---
layout: posts
title:  "Communications in Distributed Training with Tensorflow + Horovod"
author: kaixi_hou
#search                   : true
#search_full_content      : true
#search_provider          : google
#comments: true
---
(Draft, not finished)
## Introduction
Horovod is an open source toolkit for distributed deep learning when the models'
size and data consumption are too large. Horovod exhibits many benefits over the
standard distributed techniques provided by Tensorflow. The official document
has already shown that only a couple of steps can allow users to enjoy the
simplicity of training models at scale. This post, by contrast, focuses on
explaining what happens over the model parameters and their gradients during
training with TF + Horovod.

## A Simple Example
For illustration purpose, we use a one-dense-layer model which takes in a 4x2
input tensor and outputs a 4x3 tensor. Therefore, the total parameters of
interest are 2x3 weights and 3 biases. We also adopt a typical SGD optimizer and
set the learning rate as 1.0 so that the updating formula is simply `new_weights
= old_weights + gradients`. The following python script shows this model with
the settings of Horovod. To mimic the real-world scenarios, we intentially
generate different inputs and parameters for different nodes.

```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Setup of Horovod
import horovod.tensorflow as hvd
hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
  tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

rows = 4
cols = 2
units = 3

# Mimic different inputs for each node.
tf.random.set_seed(hvd.rank())
np.random.seed(hvd.rank())

x = np.random.random([rows, cols]).astype(np.float32)

# Mimic different initial parameters for each node.
dense = layers.Dense(units,
                     kernel_initializer='ones' if hvd.rank() == 0 else 'zeros',
                     bias_initializer='ones' if hvd.rank() == 0 else 'zeros')

opt = tf.optimizers.SGD(1.0)
```

## Forward Pass
We trigger a single step of training and take a look at what happens under the hood.
```python
with tf.GradientTape() as t:
  y = dense(x)
  loss = tf.reduce_sum(y)
  print("Loss", loss)
  print("Weights", dense.get_weights())
```
Suppose we execute the script with `horovodrun -np 2 python demo.py`, meaning
two nodes/GPUs are used. The outputs are below and as expected, the parameters
are initialized differently on the two nodes and we get two different loss
values. For the forward pass, no communication is needed.
```
[1,0]:Loss 
[1,0]:tf.Tensor(26.431675, shape=(), dtype=float32)
[1,0]:Weights 
[1,0]:[array([[1., 1., 1.],
[1,0]:        [1., 1., 1.]], dtype=float32),
[1,0]: array([1., 1., 1.], dtype=float32)]
[1,1]:Loss 
[1,1]:tf.Tensor(0.0, shape=(), dtype=float32)
[1,1]:Weights 
[1,1]:[array([[0., 0., 0.],
[1,1]:        [0., 0., 0.]], dtype=float32),
[1,1]: array([0., 0., 0.], dtype=float32)]
```

## Backward Pass
Before computing the gradients, we wrap the gradient tape with
`hvd.DistributedGridentType()` as:
```python
t = hvd.DistributedGradientTape(t)

grads = t.gradient(loss, dense.trainable_variables)
print("Grads", grads)
```
The output gradients are like the following and they are same between the two
nodes since Horovod performs an all-reduce communication over the local
gradients.
```
[1,0]:Grads 
[1,0]:[<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
[1,0]:    array([[1.3814857, 1.3814857, 1.3814857],
[1,0]:           [2.129148 , 2.129148 , 2.129148 ]], dtype=float32)>,
[1,0]: <tf.Tensor: shape=(3,), dtype=float32, numpy=
[1,0]:    array([4., 4., 4.], dtype=float32)>]
[1,1]:Grads 
[1,1]:[<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
[1,1]:    array([[1.3814857, 1.3814857, 1.3814857],
[1,1]:           [2.129148 , 2.129148 , 2.129148 ]], dtype=float32)>,
[1,1]: <tf.Tensor: shape=(3,), dtype=float32, numpy=
[1,1]:    array([4., 4., 4.], dtype=float32)>]
```
If the `hvd.DistributedGridentType()` line is deleted, we are able to see the
calculated local gradients before the all-reduce as below. Apparently, the
above gradients are mean gradients from all participant nodes. 
```
[1,0]:Grads
[1,0]:[<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
[1,0]:    array([[2.0128188, 2.0128188, 2.0128188],
[1,0]:           [2.7977395, 2.7977395, 2.7977395]], dtype=float32)>,
[1,0]: <tf.Tensor: shape=(3,), dtype=float32, numpy=
[1,0]:    array([4., 4., 4.], dtype=float32)>]
[1,1]:Grads
[1,1]:[<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
[1,1]:    array([[0.75015247, 0.75015247, 0.75015247],
[1,1]:           [1.4605565 , 1.4605565 , 1.4605565 ]], dtype=float32)>,
[1,1]: <tf.Tensor: shape=(3,), dtype=float32, numpy=
[1,1]:    array([4., 4., 4.], dtype=float32)>]
```
Under the hood, the communication is often asynchronized with the backward
computation when GPUs are available. In addition, the NCCL all-reduce will be
used if it is installed. To further boost the performance, Horovod will send
batches of tensors between some predefined intervals rather than performing
communication everytime when a tensor is ready in order to reduce launching
overhead. Also, small tensors might be fused to bigger ones before
communication. Please check `HOROVOD_CYCLE_TIME` and `HOROVOD_FUSION_THRESHOLD`
for more information.

After the backward pass, each node keeps the same gadients and then we update
the parameters. 
```python
opt.apply_gradients(zip(grads, dense.trainable_variables))
print("Updated Weights", dense.get_weights())
```
```
[1,0]:Updated Weights
[1,0]:[array([[-0.3814857, -0.3814857, -0.3814857],
[1,0]:        [-1.129148 , -1.129148 , -1.129148 ]], dtype=float32),
[1,0]: array([-3., -3., -3.], dtype=float32)]
[1,1]:Updated Weights
[1,1]:[array([[-1.3814857, -1.3814857, -1.3814857],
[1,1]:        [-2.129148 , -2.129148 , -2.129148 ]], dtype=float32),
[1,1]: array([-4., -4., -4.], dtype=float32)]
```
We can see the updated parameters from the two nodes are different since the
initial parameters are different in the first place. To make sure all the nodes
start from the same states, we can (1) initialize the params to be same values
in all nodes or (2) broadcast the updated params after this first step of
training. To do (1), for example, we could limit the param initializers to use
the same seeds in all nodes. However, (1) might be tricky to realize in practice
especially when the model become complex. By contrast, (2) is more achievable
and we only need to conduct a broadcast after the first train step, like:
```python
hvd.broadcast_variables(dense.variables, root_rank=0)
print("Broadcast Weights", dense.get_weights())

```
After the broadcast, all the nodes maintain the same params with the first node
(Theoretically, we could broadcast params from any participant node.).  Then,
the subsequent train steps won't need the broadcast anymore thanks to the
gradient accumulations. Similarly, the
broadcast communication will use NCCL if it is available.
```
[1,0]:Broadcast Weights
[1,0]:[array([[-0.3814857, -0.3814857, -0.3814857],
[1,0]:        [-1.129148 , -1.129148 , -1.129148 ]], dtype=float32),
[1,0]: array([-3., -3., -3.], dtype=float32)]
[1,1]:Broadcast Weights
[1,1]:[array([[-0.3814857, -0.3814857, -0.3814857],
[1,1]:        [-1.129148 , -1.129148 , -1.129148 ]], dtype=float32),
[1,1]: array([-3., -3., -3.], dtype=float32)]
```

## Reference
* [Meet Horovod: Uber’s Open Source Distributed Deep Learning Framework for TensorFlow](https://eng.uber.com/horovod/)
* [Horovod Tensor Fusion](https://horovod.readthedocs.io/en/stable/tensor-fusion_include.html)
