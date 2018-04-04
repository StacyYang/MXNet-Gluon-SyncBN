import threading

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import autograd, test_utils, autograd
from mxnet.ndarray import NDArray
from mxnet.gluon import HybridBlock, Block

__all__ = ['ModelDataParallel', 'BatchNorm']

class ModelDataParallel(Block):
    """Data parallelism

    Inputs and outputs are both list of NDArrays in different contexts.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.

    Parameters
    ----------
    module : object
        Network to be parallelized.
    ctx : list
        A list of contexts to use.
    Inputs:
        - **data**: input tensor with arbitrary shape.
    Outputs:
        - **out**: output tensor with the same shape as `data`.


    Example::

        >>> net = ModelDataParallel(model, ctx=[mx.gpu(0), mx.gpu(1)])
        >>> y = net(x)
    """
    def __init__(self, module, ctx):
        super(ModelDataParallel, self).__init__()
        self.ctx = ctx
        module.collect_params().reset_ctx(ctx = ctx)
        self.module = module

    def forward(self, inputs):
        return _parallel_apply(self.module, inputs)


class BatchNorm(HybridBlock):
    """Cross-GPU Synchronized Batch normalization (SyncBN)
    Normalizes the input with a mini-batch, then scale and shift.

    Parameters
    ----------
    axis : int, default 1
        The axis that should be normalized. This is typically the channels
        (C) axis. For instance, after a `Conv2D` layer with `layout='NCHW'`,
        set `axis=1` in `BatchNorm`. If `layout='NHWC'`, then set `axis=3`.
    momentum: float, default 0.9
        Momentum for the moving average.
    epsilon: float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    center: bool, default True
        If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: bool, default True
        If True, multiply by `gamma`. If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    use_global_stats: bool, default False
        If True, use global moving statistics instead of local batch-norm. This will force
        change batch-norm into a scale shift operator.
        If False, use local batch-norm.
    beta_initializer: str or `Initializer`, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer: str or `Initializer`, default 'ones'
        Initializer for the gamma weight.
    moving_mean_initializer: str or `Initializer`, default 'zeros'
        Initializer for the moving mean.
    moving_variance_initializer: str or `Initializer`, default 'ones'
        Initializer for the moving variance.
    in_channels : int, default 0
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    nGPUs : int, default number of visible GPUs
    Inputs:
        - **data**: input tensor with arbitrary shape.
    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, momentum=0.9, epsilon=1e-5, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 running_mean_initializer='zeros', running_variance_initializer='ones',
                 in_channels=0, nGPUs=None, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self._kwargs = {'eps': epsilon, 'momentum': momentum,
                        'fix_gamma': not scale}
        if in_channels != 0:
            self.in_channels = in_channels
        self.eps = epsilon
        self.momentum =  momentum

        self.gamma = self.params.get('gamma', grad_req='write' if scale else 'null',
                                     shape=(in_channels,), init=gamma_initializer,
                                     allow_deferred_init=True,
                                     differentiable=scale)
        self.beta = self.params.get('beta', grad_req='write' if center else 'null',
                                    shape=(in_channels,), init=beta_initializer,
                                    allow_deferred_init=True,
                                    differentiable=center)
        self.running_mean = self.params.get('running_mean', grad_req='null',
                                            shape=(in_channels,),
                                            init=running_mean_initializer,
                                            allow_deferred_init=True,
                                            differentiable=False)
        self.running_var = self.params.get('running_var', grad_req='null',
                                           shape=(in_channels,),
                                           init=running_variance_initializer,
                                           allow_deferred_init=True,
                                           differentiable=False)
        if nGPUs is None:
            nGPUs = self._get_nGPUs()
        self.xsum = SharedTensor(nGPUs)
        self.xsqu = SharedTensor(nGPUs)
        self.updater = SharedUpdater(nGPUs)

    def _get_nGPUs(self):
        # caution: if not using all the GPUs, please mannually set nGPUs
        nGPUs = len(test_utils.list_gpus())
        # for CPU
        nGPUs = nGPUs if nGPUs > 0 else 1
        return nGPUs

    def cast(self, dtype):
        if np.dtype(dtype).name == 'float16':
            dtype = 'float32'
        super(BatchNorm, self).cast(dtype)

    def hybrid_forward(self, F, x, gamma, beta, running_mean, running_var):
        if autograd.is_training():
            isum, isqu = nd.SumSquare(x)
            # reduce sum for E(x) and E(x^2)
            idsum = self.xsum.push(isum)
            idsqu = self.xsqu.push(isqu)
            osum = self.xsum.get(idsum)
            osqu = self.xsqu.get(idsqu)
            assert(len(self.xsum) == len(self.xsqu))
            N = len(self.xsum)*x.shape[0]*x.shape[2]*x.shape[3]
            # calc mean and std
            mean = osum / N
            sumvar = osqu - osum * osum / N
            std = nd.sqrt(sumvar / N + self.eps)
            # update running mean and var
            with autograd.pause():
                unbias_var = sumvar / (N - 1)
                ctx = x.context
                self.updater(self.running_mean, self.running_var, mean, unbias_var,
                             self.momentum, ctx)
            return nd.DecoupleBatchNorm(x, gamma, beta, mean, std,
                                        name='fwd', **self._kwargs)
        else:
            ctx = x.context
            return nd.BatchNorm(x, gamma, beta, running_mean, running_var, name='fwd', 
                               **self._kwargs)

    def __repr__(self):
        s = '{name}({content}'
        in_channels = self.gamma.shape[0]
        s += ', in_channels={0}'.format(in_channels if in_channels else None)
        s += ')'

        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))


def _parallel_apply(module, inputs, kwargs_tup=None):
    if kwargs_tup:
        assert len(inputs) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(inputs)

    if isinstance(inputs, NDArray):
        return module(inputs, **kwargs_tup[0])
    if len(inputs) == 1:
        return (module(inputs[0], **kwargs_tup[0]), )

    lock = threading.Lock()
    results = {}

    def _worker(i, module, input, kwargs, results, is_training, lock):
        try:
            if is_training:
                with autograd.record():
                    output = module(input, **kwargs)
                    output.wait_to_read()
            else:
                output = module(input, **kwargs)
                output.wait_to_read()
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    if autograd.is_training():
        is_training = True
    else:
        is_training = False

    threads = [threading.Thread(target=_worker,
                                args=(i, module, input, kwargs, results, 
                                      is_training, lock),
                                )
               for i, (input, kwargs) in
               enumerate(zip(inputs, kwargs_tup))]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    outputs = []
        
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


class SharedUpdater:
    # update only once
    def __init__(self, nGPUs):
        self.mutex = threading.Lock()
        self.nGPUs = nGPUs
        self._clear()

    def _clear(self):
        self.tasks = self.nGPUs

    def __call__(self, running_mean, running_var, mean, unbias_var, momentum, ctx):
        with self.mutex:
            if self.tasks == self.nGPUs:
                running_mean.set_data(momentum * running_mean.data(ctx) + \
                    (1.0 - momentum) * mean)
                running_var.set_data(momentum * running_var.data(ctx) + \
                    (1.0 - momentum) * unbias_var)
            self.tasks -= 1
        if self.tasks == 0:
            self._clear()

class SharedTensor:
    def __init__(self, nGPUs):
        self.mutex = threading.Lock()
        self.all_tasks_done = threading.Condition(self.mutex)
        self.nGPUs = nGPUs
        self._clear()

    def _clear(self):
        self.list = []
        self.push_tasks = self.nGPUs
        self.reduce_tasks = self.nGPUs

    def push(self, t):
        with self.mutex:
            if self.push_tasks == 0:
                self._clear()
            self.list.append(t)
            idx = len(self.list) - 1
            self.push_tasks -= 1

        with self.all_tasks_done:
            if self.push_tasks == 0:
                self.all_tasks_done.notify_all()
            while self.push_tasks:
                self.all_tasks_done.wait()
        return idx

    def _reduce(self):
        with self.mutex:
            if self.reduce_tasks == 1:
                assert(len(self.list) == self.nGPUs)
                self.list = nd.AllReduce(*self.list)
                for xi in self.list:
                    # mannually attach grad to avoid wrong allocation
                    xi.attach_grad()
                    xi.wait_to_read()
                self.reduce_tasks -= 1
            else:
                self.reduce_tasks -= 1

        with self.all_tasks_done:
            if self.reduce_tasks == 0:
                self.all_tasks_done.notify_all()
            while self.reduce_tasks:
                self.all_tasks_done.wait()

    def get(self, idx):
        self._reduce()
        return self.list[idx]

    def test(self):
        print('self.list', self.list)

    def __len__(self):
        return len(self.list)

    def __repr__(self):
        return 'SharedTensor'
