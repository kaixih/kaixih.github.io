---
layout: posts
title:  "Moving Mean and Moving Variance In Batch Normalization"
published: false
author: kaixi_hou
search                   : true
search_full_content      : true
search_provider          : google
comments: true
---

## Introduction
My previous post xxx has talked about differnt types of normalizations. which
all conist two part, statistics computing and normalization. Their computation
over different axis according to norm types. In them,
batch normalization might be the most different one, which needs to computattion
accross the batch axis, and importantly, batch normalization works differently
during training and during inference. While working on the backend of BN, I hear of
things moving mean, batch mean, estimated mean, and even saved mean. They are sometimes
confusing and in this post I want to talk about the difference of them and show
how they are used. This will cover some Tensorflow karas batch norm layer and
its backend CUDNN bactch norm APIs.

## Typical Batch Norm
In the typical batch norm, the moments will be first called to calculate the
_*batch mean/variance*_ (or _*corrent mean/variance*_, _*new mean/variance*_, etc.) it is only reflect the
current input x's statistics. As shown in Figure 1, the m' and v' are them.
Then, they are fed into update op to update the _*moving mean/variance*_ (or
_*running mean/variance*_). Update Op's formula is like `moving_xxx = moving_xxx *
momentum + batch_mean * (1 - momentum)`. Momentum is a hyperparam. It worth to
mention in cudnn we use `expoentialaverge_factor` which is simply
`1-momentum`. So the formula like
`moving_xx=moving_xx(1-factor)+batch_mean(factor)`. The normalization Op,
however, will still use the batch_mean/variance m' and v' to do the computation.

<p align=center> Figure 1. Typical batch norm in Tensorflow Keras</p>
![Typical Batch Norm](/assets/posts_images/bn_orig.png)

Here I devise an example to minic a one-step training and check how the moving
mean/var are changed. We use a single bn layer model and ignore the offset/scale
update. It worth to note that the moving mean/var are non-trainable variable of
the layer which will be accumulate during training and will use used in
inference.
```python
bn = tf.keras.layers.BatchNormalization(momentum=0.5,
    beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='ones', moving_variance_initializer='ones',
    fused=False)
# Training Step 0 (skip the real scale/offset update)
x0 = tf.random.uniform((2,1,2,3))
print("x(step0):", x0.numpy())

y0 = bn(x0, training=True)
print("moving_mean(step0): ", bn.moving_mean.numpy())
print("moving_var(step0): ", bn.moving_variance.numpy())
print("y(step0):", y0.numpy())
```
The output of above code is like:
```
x(step0):
[[[[0.16513085 0.9014813  0.6309742 ]
   [0.4345461  0.29193902 0.64250207]]]
 [[[0.9757855  0.43509948 0.6601019 ]
   [0.60489583 0.6366315  0.6144488 ]]]]
!!! batch_mean:
   [0.5450896  0.5662878  0.63700676]
!!! batch_var: 
   [0.08641606 0.05244513 0.00027721]
moving_mean(step0):
   [0.7725448 0.7831439 0.8185034]
moving_var(step0):
   [0.543208   0.5262226  0.50013864]
y(step0):
[[[[-1.2851115   1.4499114  -0.16879845]
   [-0.37388456 -1.186722    0.15376663]]]
 [[[ 1.4567168  -0.5674678   0.6462345 ]
   [ 0.20227909  0.30427837 -0.6312027 ]]]]
```
Note the batch_mean/var are some injected print code to get from internal.
First, since we set the init moving_mean/var to be ones, and momentue to be 0.5,
by using moving_xx = 0.5+0.5batch_mean we can get batch_xxx(step0). Second, it
is easy to confirm the
y(step0) is use x-batch_mean/sqrt(batch_var+0.001).

In real case, we usually need to repeat the step many round. For simplicity, we
can conduct a second round the these trianing with another round of fake data
and get sth like
```
moving_mean(step1):
    [0.72269297 0.63172865 0.62922215]
moving_var(step1):
    [0.29549128 0.28121024 0.25281417]
```
Then we do an inference:
```python
# Inference Step
x_infer = tf.random.uniform((2,1,2,3))
print("x(infer):", x_infer.numpy())

y_infer = bn(x_infer, training=False)
print("estimated_mean(infer): ", bn.moving_mean.numpy())
print("estimated_var(infer): ", bn.moving_variance.numpy())
print("y(infer):", y_infer.numpy())
```
Here for the _*estimizedmean/var*_, they are accumulated moving_mean/var from the
training stage. Or we can say moving_mean are evolved into estimated_mean at the
end of training.
We can see below that the y(infer) is using the estimazted mean/var rather than
batch_mean/var: x-estimated_mean/sqrt(estimated_var+0.001). And the estimated_mean will no longer updated which is still
same with movingmean(step1).
```
x(infer):
[[[[0.8292774  0.634228   0.5147276 ]
   [0.39108086 0.5809028  0.04848182]]]
 [[[0.1776321  0.70470166 0.49190843]
   [0.3460425  0.5079107  0.2216742 ]]]]
!!! batch_mean:
   [0.72269297 0.63172865 0.62922215]
!!! batch_var:
   [0.29549128 0.28121024 0.25281417]
estimated_mean(infer):
   [0.72269297 0.63172865 0.62922215]
estimated_var(infer):
   [0.29549128 0.28121024 0.25281417]
y(infer):
[[[[ 0.1957438   0.00470471 -0.22726214]
   [-0.60901    -0.09567499 -1.1527207 ]]]
 [[[-1.0010114   0.13736498 -0.2725562 ]
   [-0.69172347 -0.2330758  -0.8089484 ]]]]
```
The full code is
[here](https://github.com/kaixih/dl_samples/blob/main/batch_norm/tf_keras_bn.py).

## Fused Batch Norm
In the above example we use fused=False, which will explictly turn off using
fused kernel. In practive, we usually leave it as None (use fusion when
possible) or True(force to use fusion) for better
performance. Figure 2 shows what happens when fussion is applied. We can see
there is only one big kernel in BN `FusedBatchNorm`. The inputs/outputs are same
with previous example, however, we cannot easily get the m' and v', which should
be fine in most cases except for the sync batch norm that we will cover later.
Note by turn on the fusion the resule are no guarantee of bitwise equality with
above.
<p align=center> Figure 2. Fused batch norm on GPUs</p>
![Fused Batch Norm](/assets/posts_images/bn_fuse.png)

## Batch Norm Backpropagation
If one work with cudnn libary, we might came accross another term _*saved mean and
saved inv variance*_, they are specially for the backpropagation. It usually hid
from users. Therefore, if
we apply something like we do a complete forward and backward pass. Note we
still didn't update scale and offset. Its op graph will be liek Figure 3.
```python
bn = tf.keras.layers.BatchNormalization(momentum=0.5,
    beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='ones', moving_variance_initializer='ones',
    fused=True)
x0 = tf.random.uniform((2,1,2,3))
with tf.GradientTape() as t:
  t.watch(x0)
  y0 = bn(x0, training=True)
  loss = tf.reduce_sum(y0)
grads = t.gradient(loss, [x0, bn.trainable_variables])
```
In Figure 3, we notice that the backward pass needs dy, and g(scale) and also r1
r2, r3. For the CUDNN, r1 and r2 are saved mean and
saved inv variance respectively. In fact they are two optional paramters for
CUDNN for performance, if not given, CUDNN has to re-compute them in the
backward pass. In TF, they are always set.
<p align=center> Figure 3. Fused batch norm and backpropagation</p>
![Batch Norm Backpropagation](/assets/posts_images/bn_grad.png)
It hard to get saved mean/inv var from keras API. fortunately, we can directly
use the op from tf.raw ops to peek what's in them:
```python
y, batch_mean, batch_var, r1, r2, r3 = tf.raw_ops.FusedBatchNormV3(
    x=x, scale=scale, offset=offset, mean=mean, variance=variance,
    epsilon=0.001, exponential_avg_factor=0.5, data_format='NHWC',
    is_training=True, name=None)
```
And by using the same input as exmaple 1, we can get following outputs. The
moving mean/var are similar to example 1 after 1st step (They are not bitwise
same since we use fusion here). Saved mean is basically same with batch_mean and
the saved in var are `1/sqrt(batch_var+0.001)`.
```
moving_mean: [0.7725448 0.7831439 0.8185034]
moving_var: [0.5576107  0.5349634  0.50018483]
saved mean: [0.5450896  0.5662878  0.63700676]
saved inv var: [ 3.3822396  4.325596  27.981367 ]
```
The full use of
[backpropagation](https://github.com/kaixih/dl_samples/blob/main/batch_norm/tf_keras_bn_grad.py) and
[raw_ops](https://github.com/kaixih/dl_samples/blob/main/batch_norm/tf_bn_raw_ops.py).
The backend of V3 are using CUDNN calls. Here is an example of to do the same
thing using pure c++,
[cudnn](https://github.com/kaixih/dl_samples/blob/main/batch_norm/cudnn_batch_norm.cu).
which we can test the saved mean/inv variance are not required parameters.

## Synchronized Batch Norm
Last, I would like to talk about the sync batch norm which are beneficial when
the batch size are too small for each node. Here I use a implement
from horovod synch batch norm, which overrides the moment op to return the _*group
mean and var*_ and then the norm op can correctly use the group mean and var.
Therefore the m' and v' are group mean and variance and moving_mean/var and more precisely moving_group_mean as shown in Figure 4. Since its rely on the moment, it doesn't support the fused kernel.
<p align=center> Figure 4. Synchronized batch norm</p>
![Synchronized Batch Norm](/assets/posts_images/bn_sync.png)
Here is an example:
```python
# Make sure that different ranks have different inputs.
tf.random.set_seed(hvd.local_rank())
x0 = tf.random.uniform((2,1,2,3))

sync_bn = hvd.SyncBatchNormalization(axis=-1, momentum=0.5,
    beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='ones', moving_variance_initializer='ones')
print("moving_mean: ", sync_bn.moving_mean.numpy())
print("moving_var: ", sync_bn.moving_variance.numpy())
```
If we run is with two nodes `horovodrun -np 2 python tf_hvd_bn_sync.py`.
```
[1,1]:!!! batch_mean:[0.45652372 0.5745423  0.65629953]
[1,0]:!!! batch_mean:[0.45652372 0.5745423  0.65629953]
[1,1]:!!! batch_var:[0.0657616  0.0710133  0.00523123]
[1,0]:!!! batch_var:[0.0657616  0.0710133  0.00523123]
[1,0]:moving_mean:[0.7282618  0.78727114 0.8281498 ]
[1,1]:moving_mean:[0.7282618  0.78727114 0.8281498 ]
[1,0]:moving_var:[0.5328808  0.53550667 0.50261563]
[1,1]:moving_var:[0.5328808  0.53550667 0.50261563]
```
As we can see, the batch mean/var are already group mean/var after sync and the
moving/meanvar are called by it. One more thing needs to mention, the
communication dot lines in Figure 4 might be not precisely what happened under
the hood. The group
mean are average of batch mean from different ranks which can be done with
allreduce. But that is not the case
for the variance. The group variance cannot averge batch variance from different
  ranks rather it needs to recompute using the newly computed group_mean
(x-group_mean)^2/nxbatch_size, the n is number of ranks.

The  full code is
[here](https://github.com/kaixih/dl_samples/blob/main/batch_norm/tf_hvd_bn_sync.py).

## Reference
* [Tensorflow Batch Normalization API](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)
* [Deriving the Gradient for the Backward Pass of Batch Normalization](https://kevinzakka.github.io/2016/09/14/batch_normalization/)
* [CUDNN Batch Normalization API](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnBatchNormalizationForwardTraining)
* [Add SyncBatchNormalization layer for TensorFlow](https://github.com/horovod/horovod/pull/2075)


