MXNet-Gluon-SyncBN
==================
`Hang Zhang <http://hangzh.com/>`_

A preview tutorial for MXNet Gluon Synchronized Batch Normalization (SyncBN)[1]_. We follow the sync-onece implmentation described in the paper[2]_. If you are not familiar with Synchronized Batch Normalization, please see this `blog <http://hangzh.com/SynchronizeBN/>`_.

Jump to:
- `How to use Synchronized Batch Normalization`_
- `MNIST example <https://github.com/zhanghang1989/MXNet-Gluon-SyncBN/blob/master/mnist.ipynb>`_
- `Load Pre-trained Network`_

Install MXNet from Source
-------------------------

* Please follow the MXNet docs to `install dependencies <http://mxnet.incubator.apache.org/install/index.html>`_
* Clone and install `syncbn branch <https://github.com/zhanghang1989/incubator-mxnet/tree/syncbn>`_::

    # clone the branch
    git clone https://github.com/zhanghang1989/incubator-mxnet
    git checkout syncbn
    # compile mxnet
    make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
    # install python API
    cd python && python setup.py install

How to use SyncBN
-----------------

``from syncbn import BatchNorm`` and everything else looks the same as before::

    import mxnet as mx
    from mxnet import gluon, autograd
    from mxnet.gluon import nn
    from mxnet.gluon.nn import Block

    from syncbn import BatchNorm, ModelDataParallel

    # create your own Block
    class Net(Block):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2D(in_channels=3, channels=10,
                                  kernel_size=3, padding=1)
            self.bn = BatchNorm(in_channels=10)
            self.relu = nn.Activation('relu')

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x.wait_to_read()
            return x

    # set the contexts (suppose using 4 GPUs)
    nGPUs = 4
    ctx_list = [mx.gpu(i) for i in range(nGPUs)]
    # get the model
    model = Net()
    model.initialize()
    model = ModelDataParallel(model, ctx_list)
    # load the data
    data = mx.random.uniform(-1,1,(8, 3, 24, 24))
    x = inputs = gluon.utils.split_and_load(data, ctx_list=ctx_list)
    with autograd.record():
        y = model(x)


**Note**: you have to use ModelDataParallel to do network forward (input and output are both a list of NDArray).

MNIST Example
-------------

Please visit the `python notebook <https://github.com/zhanghang1989/MXNet-Gluon-SyncBN/blob/master/mnist.ipynb>`_

Load Pre-trained Network
------------------------

**TODO**

Reference
---------

.. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." *ICML 2015*

.. [2] Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, and Amit Agrawal. "Context Encoding for Semantic Segmentation." *CVPR 2018*
