---
layout: posts
title:  "Inside Normalizations of Tensorflow"
author: kaixi_hou
search                   : true
search_full_content      : true
search_provider          : google
comments: true
---
## Introduction
Recently I came across with optimizing the normalization layers in Tensorflow.
Most online articles are talking about the mathematical definitions of different 
normalizations and their advantages over one another. Assuming that you have
adequate background of these norms, in this blog post, I'd like to provide a
practical guide to using the relavant norm APIs from Tensorflow, and give you an
idea when the fast CUDNN kernels will be used in the backend on GPUs.

This post will only checks the **BatchNorm**, **LayerNorm**, and
**InstanceNorm**. In essence, all these norms perform a 2-step calculation:
1. Computing mean and variance (also called statistics, moments, etc.);
2. Applying scale and offset (a.k.a gamma/beta, which are two learnable
parameters).

The trickly part is that the axis values and output
shapes from (1) and (2) vary depending on normalization types and sometimes the
official API document might be misleading. Therefore, I am going to review how
to use these three norm APIs in practice and what happens under the hood.

Note: the sample codes below use BatchNormalization and LayerNormalization from
TF Keras Layers and InstanceNormalization from TF Addons.

## Batch Normalization
Let's start with an example tensor in shape of (2, 12, 3, 2) and its format is
NCHW (or "channels_first"), meaning there are 12 channels and its axis is 1.
BatchNorm expects the `axis` argument to be channels axis and thus we can put 1
here. Under the hood, the API will perform step (1) and (2) along the same axis
and you will get the mean/var in shape of (1, 12, 1, 1) and the scale/offset in
shape of (12,). Thanks to the broadcasting rules, the step (2) can be easily
implemented. It is also for this reason that `nn.batch_normalization` is used as
the backend in other types of normalization.  Instead of this "generic"
`nn.batch_normalization`, the backend will call the faster CUDNN API, e.g.,
`cudnnBatchNormalizationForwardTraining()` whenever possible (e.g., the data
type and axis fulfill some requirements and of cause the GPU can be detected) so
that we can benefit from its efficient fused parallel kernels and reduced memory
footprint.

The following example checks the shape of gamma/beta and verify if the mean/var
are computed along the given axis.
```python
  batch_norm = layers.BatchNormalization(axis=1, center=True, scale=True)
  y = batch_norm(x, training=True)
  print("Gamma shape:", batch_norm.weights[0].shape) # Output: (12,)
  print("Beta  shape:", batch_norm.weights[1].shape) # Output: (12,) 
  may_pass = True
  for i in range(C):
    if not np.isclose(tf.math.reduce_mean(y[:, i, ...]).numpy(), 0.0,
                      rtol=1e-06, atol=1e-06):
      may_pass = False
  print("Test:", "Pass!" if may_pass else "Fail!")
```

Similaly, the axis argument should take -1 or 3 when the NHWC (or
"channels_last") is used.

## Layer Normalization
Continuing with the same example tensor above, LayerNorm usually expects the
`axis` argument to take in the features within one sample; hence, we must not
include the batch axis. Here one legit `axis` is (1,2,3), meaning we include all
features for each sample. Under the hood, the computed mean/var will be in shape
of (2, 1, 1, 1) and the scale/offset in (12, 3, 2). The "generic"
`nn.batch_normalization` has no problem to realize the step (2) due to the
broadcasting rules. However, as for the CUDNN APIs, they lack the support for
layer norm and even worse we cannot directly call its batch norm APIs since this
computational pattern breaks the CUDNN's assumption that the two shapes of step
(1) and (2) should be same. Thus, TF works it around with a two-step
implementation: First, call CUDNN with a dummy scale/offset in the same shape of
mean/var but filled with 1s and 0s; Second, apply the real scale/offset in shape
(12,3,2). Though this doesn't fully benefit from the CUDNN kernels, it is better
than none.

The following example checks the shape of gamma/beta and verify if the mean/var
are computed along the given axes.
```python
  layer_norm = layers.LayerNormalization(axis=(1,2,3), center=True, scale=True)
  y = layer_norm(x)
  print("Gamma shape:", layer_norm.weights[0].shape) # Output: (12, 3, 2)
  print("Beta  shape:", layer_norm.weights[1].shape) # Output: (12, 3, 2)
  may_pass = True
  for i in range(N):
    if not np.isclose(tf.math.reduce_mean(y[i,...]).numpy(), 0.0,
                      rtol=1e-06, atol=1e-06):
      may_pass = False
  print("Test:", "Pass!" if may_pass else "Fail!")
```

Similaly, for the NHWC tensors, the axis argument can take the same (1,2,3) as
in the NCHW use case.

## Instance Normalization
In InstanceNorm, the expected axis is same with BatchNorm, i.e. the
channels axis. So, for the same example above, we would set axis=1. Internally,
however, the batch axis will also be considered to compute the mean/var,
producing the output in shape of (2, 12, 1, 1). On the other hand, the
scale/offset will still be (12,). So, it would be tricky to put this
computational pattern under the disguise of the batch norm as we do in layer
norm (because we need to deal with two non-singleton dimensions in the
mean/var and this apparently fails to follow the CUDNN's assumption).

The following example checks the shape of gamma/beta and verify if the mean/var
are computed along the given axes.
```python
  instance_norm = tfa.layers.InstanceNormalization(axis=1, center=True,
                                                   scale=True)
  y = instance_norm(x)
  print("Gamma shape:", instance_norm.weights[0].shape) # Output: (12,)
  print("Beta  shape:", instance_norm.weights[1].shape) # Output: (12,)
  may_pass = True
  for i in range(N):
    for j in range(C):
      if not np.isclose(tf.math.reduce_mean(y[i,j,...]).numpy(), 0.0,
                        rtol=5e-06, atol=5e-06):
        print(tf.math.reduce_mean(y[i,j,...]).numpy())
        may_pass = False
  print("Test:", "Pass!" if may_pass else "Fail!")
```

Similaly, the axis argument should take -1 or 3 when the NHWC is used.


## Group Normalization
In GroupNorm, the axis should also be set to channels. Besides, we can also
split the channels into different groups and the mean/var
computation will be within each groups. So, for the same example above, if we
set the `axis=1` and `group=4`, the input tensor will be reshaped to
(2, 4, 3, 3, 2) and the mean/var will be (2, 4, 1, 1, 1). The scale/offset will
stay in the shape of (12,).

```python
G = 4
group_norm = tfa.layers.GroupNormalization(groups=G, axis=1, center=True,
                                           scale=True)
y = group_norm(x)
print("Gamma shape:", group_norm.weights[0].shape) # Output: (12,)
print("Beta  shape:", group_norm.weights[1].shape) # Output: (12,)
y = tf.reshape(y, shape=(N, G, C // G, H, W))
print("Reshape by grouping:", y.shape) # Output: (2, 4, 3, 3, 2)
may_pass = True
for i in range(N):
  for j in range(G):
    if not np.isclose(tf.math.reduce_mean(y[i,j,...]).numpy(), 0.0,
                      rtol=5e-06, atol=5e-06):
      print(tf.math.reduce_mean(y[i,j,...]).numpy())
      may_pass = False
print("Test:", "Pass!" if may_pass else "Fail!")
```

## Reference
* [Tensorflow Batch Normalization API](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)
* [Tensorflow Layer Normalization API](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)
* [Tensorflow Instance Normalization API](https://www.tensorflow.org/addons/api_docs/python/tfa/layers/InstanceNormalization)

