from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn
from syncbn import BatchNorm, ModelDataParallel

# Set the contexts (suppose using 4 GPUs)
nGPUs = 4
ctx_list = [mx.gpu(i) for i in range(nGPUs)]

# MNIST dataset
batch_size = 128
num_outputs = 10
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)
train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform=transform),
                                   batch_size, shuffle=True, last_batch='rollover',
                                   num_workers=4)
test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False, num_workers=4)

# Define a convolutional neural network
num_fc = 512
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Conv2D(in_channels=1, channels=20, kernel_size=5))
    net.add(BatchNorm(in_channels=20, nGPUs=nGPUs))
    net.add(gluon.nn.Activation('relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net.add(gluon.nn.Conv2D(in_channels=20, channels=50, kernel_size=5))
    net.add(BatchNorm(in_channels=50, nGPUs=nGPUs))
    net.add(gluon.nn.Activation('relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    # The Flatten layer collapses all axis, except the first one, into one axis.
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(num_fc,in_units=800))
    net.add(gluon.nn.Activation('relu'))
    net.add(gluon.nn.Dense(num_outputs, in_units=num_fc))

# Initializae the model wieghts and get Parallel mode
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24))
net = ModelDataParallel(net, ctx_list)
print(net)

# Loss and Optimizer
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

# Write evaluation loop
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = gluon.utils.split_and_load(data, ctx_list=ctx_list)
        label = gluon.utils.split_and_load(label, ctx_list=ctx_list)
        outputs = net(data)
        predictions = []
        for i, output in enumerate(outputs):
            pred = nd.argmax(output, axis=1)
            acc.update(preds=pred, labels=label[i])
    return acc.get()[1]

# start training
epochs = 5
smoothing_constant = .01

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = gluon.utils.split_and_load(data, ctx_list=ctx_list)
        label = gluon.utils.split_and_load(label, ctx_list=ctx_list)
        with autograd.record():
            output = net(data)
            losses = [softmax_cross_entropy(yhat, y) for yhat, y in zip(output, label)]
            autograd.backward(losses)
        loss = 0
        for l in losses:
            loss += l.as_in_context(mx.gpu(0))
        trainer.step(len(data)*data[0].shape[0])
        ##########################
        #  Keep a moving average of the losses
        ##########################
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))
