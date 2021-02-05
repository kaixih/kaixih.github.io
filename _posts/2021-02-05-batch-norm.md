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
On my previous post [Inside Normalizations of
Tensorflow](https://kaixih.github.io/norm-patterns/) we discussed three common
normalizations used in deep learning. They have in common a two-step
computation: (1) statistics computation to get mean and variance and (2)
normalization with scale and shift, though each step requires different
shape/axis for different normalization types. Among them, the batch
normalization might be the most special one, where the statistics computation is
performed across batches. More importantly, it works differently during training
and inference. While working on its backend optimization, I frequently
encountered various concepts describing mean and variance: moving mean, batch
mean, estimated mean, and even saved mean to name a few. Therefore, this post
will look into the differences of these terms and show you how they are used in
deep learning framework, Tensorflow Keras Layers, and deep learning library,
CUDNN Batch Norm APIs.


## Typical Batch Norm
In a typical batch norm, the "Moments" op will be first called to compute the
statistics of the input `x`, i.e. the _*batch mean/variance*_ (or _*current
mean/variance*_, _*new mean/variance*_, etc.). It only reflects the local
information of `x`. As shown in Figure 1, we use `m'` and `v'` to represent
them. After statistics computation, they are fed into the "Update" op to obtain
the new _*moving mean/variance*_ (or _*running mean/variance*_).  The formula
used here is `moving_*** = moving_*** ⋅ momentum + batch_*** ⋅ (1 - momentum)`
where the momentum is a hyperparameter. (Instead, CUDNN uses a so called
exponential average `factor` and thus its updating formula becomes `moving_*** =
moving_*** ⋅(1 - factor) + batch_*** ⋅factor`.) In the second step for
normalization, the "Normalize" op will use the batch mean/variance `m'`
and `v'` as well as the scale (gamma) and offset (beta).

<p align=center> Figure 1. Typical batch norm in Tensorflow Keras</p>
![Typical Batch Norm](/assets/posts_images/bn_orig.png)

The following script shows an example to mimic one training step of a single
batch norm layer. Tensorflow Keras API allows us to peek the moving
mean/variance but not the batch mean/variance. For illustrative purposes, I
inserts some print()s to the Keras python functions to get the batch
mean/variance. Note, the moving mean/variance are not trainable variables so
that they cannot be updated during the backpropagation. For this reason, we skip
the backward pass in the training step.

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
The outputs of the above code are like the below and we can see that the moving
mean/variance are different from batch mean/variance. Since we set the momentum
to 0.5 and the initial moving mean/variance to ones, the updated mean/variance
are calculated by `moving_*** = 0.5 + 0.5 ⋅batch_***`. On the other hand, it can
be confirmed that the `y_step0` is computed with the batch mean/variance through
`(x_step0 - batch_mean) / sqrt(batch_var + 0.001)` 
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
To make it closer to real settings, we conduct one more training step with
another input `x` and these are the moving mean/variance we can get:

```
moving_mean(step1):
    [0.72269297 0.63172865 0.62922215]
moving_var(step1):
    [0.29549128 0.28121024 0.25281417]
```
At last, we mimic one step of inference as below. In the script, we rename the
moving mean/variance to _*estimized_mean/variance*_, which represents the
accumulated and frozen moving mean/variance from the training stage.
```python
# Inference Step
x_infer = tf.random.uniform((2,1,2,3))
print("x(infer):", x_infer.numpy())

y_infer = bn(x_infer, training=False)
print("estimated_mean(infer): ", bn.moving_mean.numpy())
print("estimated_var(infer): ", bn.moving_variance.numpy())
print("y(infer):", y_infer.numpy())
```
From the outputs below, it is easy to verify that `y_infer` is computed with the
estimated mean/variance rather than batch mean/variance: `(x_infer -
estimated_mean) / sqrt(estimated_var + 0.001)`. Besides, we can see that the
estimated mean/variance equal to the moving mean/variance from the above step1
and will be no longer updated. To sum up, the takeaway here is that the batch
norm will keep accumulating batch mean/variance into the moving mean/variance
during training, which will be evolved into frozen estimated mean/variance to be
used during inference.
 
```
x(infer):
[[[[0.8292774  0.634228   0.5147276 ]
   [0.39108086 0.5809028  0.04848182]]]
 [[[0.1776321  0.70470166 0.49190843]
   [0.3460425  0.5079107  0.2216742 ]]]]
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


