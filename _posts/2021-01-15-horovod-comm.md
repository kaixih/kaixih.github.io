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
standard distributed techniques provided by Tensorflow. This post Training deep learning models 

Mainly about how the gradients and weights are communicated when training with
TF and Horovod.

For illustration purpose, use a single dense layer and focus on how
communication is done. Here is the setup, using hvd is simple: import and init
it and bind to different GPUs. After that we define a single Dense layer, which 
takes a 4x2 matrix and output 4x3 matrix. So, the total weights are 2x3 kernels
and 3 biases. we intentially the set inputs to be different for differnt ranks
to mimic the behavior that different ranks load different piece of training
data. Similarly differnet rank initializes theirs model with different
parameters. The optimizer is SGD learning rate 1.0 with no momentum or nestkov
to realize the simplest update formulate w = old_w+1xg.

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

# Mimic different inputs for each rank
tf.random.set_seed(hvd.rank())
np.random.seed(hvd.rank())

x = np.random.random([rows, cols]).astype(np.float32)

# Mimic different initial weights for each rank
dense = layers.Dense(units,
                     kernel_initializer='ones' if hvd.rank() == 0 else 'zeros',
                     bias_initializer='ones' if hvd.rank() == 0 else 'zeros')

opt = tf.optimizers.SGD(1.0)
```
Then, we start one step of training to check what happens under the hood when
multiple nodes/GPUs are used. For the forward pass:
```python
with tf.GradientTape() as t:
  y = dense(x)
  loss = tf.reduce_sum(y)
  print("Loss", loss)
  print("Weights", dense.get_weights())

```
If we run it with horovodrun -np 2, we are using 2 nodes/GPUs to train.
Apparently, the weights are correctly initialzed for the two ranks and they
compute different loss values. For the forward, no communication is needed.
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

Then, we wrap the gradient tape with DistributedGridentType and run the backward
pass as:
```python
t = hvd.DistributedGradientTape(t)

grads = t.gradient(loss, dense.trainable_variables)
print("Grads", grads)

```
The outputs are like the following. We can see that the gradients are same for
both ranks because during the gradient computation, the horovod performs all
reduce communication to accumulate the gradients from different ranks.
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
To confirm it if no t = hvd.DistributedGradientTape(t) added, we will get the
following. Appareently, the there is reduce mean applied between weights from
different nodes. Note, the communication is not necessary synchronized with
backward computation. With the GPUs, the communication is aynchronization, and
only when a time interval the communication will be done and also the
communication is not launched when any gradient tensor is ready. check this CYC TIME FUSION for more info.
If the underlying machine supports nccl, the all-reduce communcation is based on
it.
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

So, after the backward pass, we have each rank have same gadients and we apply
the updateing (the above formula) the weights. 
```python
opt.apply_gradients(zip(grads, dense.trainable_variables))
print("Updated Weights", dense.get_weights())
```
```
[1,0]:Updated Weights
[1,0]:[array([[-0.3814857, -0.3814857, -0.3814857],
[1,0]:       [-1.129148 , -1.129148 , -1.129148 ]], dtype=float32),
[1,0]: array([-3., -3., -3.], dtype=float32)]
[1,1]:Updated Weights
[1,1]:[array([[-1.3814857, -1.3814857, -1.3814857],
[1,1]:       [-2.129148 , -2.129148 , -2.129148 ]], dtype=float32),
[1,1]: array([-4., -4., -4.], dtype=float32)]
```
You can see the updated weights
are not same because the init weights are different for the two ranks. To make
sure all the ranks are on the same page, we can (1) init the weights to be
same values (2) broadcast the weights from the very first step. To do (1), for
example, we can limit the kernel init and bias init in above to be same for
different ranks and make sure they use same seeds. However, in practice, (1) is 
hard to realize when the model become complex. (2) is more simple, we only make
sure the weights are broadcast to each rank after the first train step, like.
```python
hvd.broadcast_variables(dense.variables, root_rank=0)
print("Broadcast Weights", dense.get_weights())

```
After the broadcast all the ranks will use the same weights as the first rank
(Theretically, we can broadcast any rank).
Then the following train steps don't need the broadcast anymore since they are
already on the same page and communication later is only for gradient
all-reduce. Still If the underlying machine supports nccl, the broadcast communcation is based on
NCCL.
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

