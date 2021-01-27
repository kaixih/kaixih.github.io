---
layout: posts
title:  "Demystifying the Conv-Bias-ReLU Fusion"
#published: false
author: kaixi_hou
search                   : true
search_full_content      : true
search_provider          : google
comments: true
---
(Under construction)
## Introduction
My previous post, "[Fused Operations in
Tensorflow](https://kaixih.github.io/fused-api/)", introduced the basics of
fusion operations in deep learning by showing how to enable the grappler
optimizer in Tensorflow to recognize the supported patterns and then fuse them
together for better performance. In that post, I talked about the Conv-Bias-Relu
pattern, one of the most common patterns we can find in CNN models. In this
post, by constrast, I will dive deeper into its computational patterns and
discuss why and how they can be fused.

## Convolution Pattern
Let's start from this following figure, which represents a simple convolution
that takes in a 3x3 input and 2x2 weight and generates a 2x2 output.

<p align=center> Fig 0. Convolution's Computational Pattern </p>
![Convolution Pattern](/assets/posts_images/conv_pattern.PNG)

### Convolution Forward

Convolution equations |
--- |
y<sub>11</sub> = w<sub>11</sub>x<sub>11</sub> + w<sub>12</sub>x<sub>12</sub> + w<sub>21</sub>x<sub>21</sub> + w<sub>22</sub>x<sub>22</sub> |
y<sub>12</sub> = w<sub>11</sub>x<sub>12</sub> + w<sub>12</sub>x<sub>13</sub> + w<sub>21</sub>x<sub>22</sub> + w<sub>22</sub>x<sub>23</sub> |
y<sub>21</sub> = w<sub>11</sub>x<sub>21</sub> + w<sub>12</sub>x<sub>22</sub> + w<sub>21</sub>x<sub>31</sub> + w<sub>22</sub>x<sub>32</sub> |
y<sub>22</sub> = w<sub>11</sub>x<sub>22</sub> + w<sub>12</sub>x<sub>23</sub> + w<sub>21</sub>x<sub>32</sub> + w<sub>22</sub>x<sub>33</sub> |

The above set of equations map the outputs with the inputs. In Tensorflow, we
can use a single call to `conv2d` to realize the computation.
```python
import tensorflow as tf
x = tf.reshape(tf.range(0, 9, dtype=tf.float32), (1, 3, 3, 1))
print("x:\n", x[0, :, :, 0].numpy())
w = tf.ones((2, 2, 1, 1))
print("w:\n", w[:, :, 0, 0].numpy())
y = tf.nn.conv2d(x, w, (1, 1), 'VALID', data_format='NHWC')
print("y:\n", y[0, :, :, 0].numpy())
```
```
x:
 [[0. 1. 2.]
 [3. 4. 5.]
 [6. 7. 8.]]
w:
 [[1. 1.]
 [1. 1.]]
y:
 [[ 8. 12.]
 [20. 24.]]
```

### Convolution Backward
Suppose e is the error returned by the cost/loss function and dy is equivalent
with ∂e/∂y. According to the above equations, we can get dw = ∂e/∂w =
(∂e/∂y)(∂y/∂w) = dy⋅x. More precisely, the equations for dw are:

Weight gradient equations |
--- |
dw<sub>11</sub> = dy<sub>11</sub>x<sub>11</sub> + dy<sub>12</sub>x<sub>12</sub> + dy<sub>21</sub>x<sub>21</sub> + dy<sub>22</sub>x<sub>22</sub> |
dw<sub>12</sub> = dy<sub>11</sub>x<sub>12</sub> + dy<sub>12</sub>x<sub>13</sub> + dy<sub>21</sub>x<sub>22</sub> + dy<sub>22</sub>x<sub>23</sub> |
dw<sub>21</sub> = dy<sub>11</sub>x<sub>21</sub> + dy<sub>12</sub>x<sub>22</sub> + dy<sub>21</sub>x<sub>31</sub> + dy<sub>22</sub>x<sub>32</sub> |
dw<sub>22</sub> = dy<sub>11</sub>x<sub>22</sub> + dy<sub>12</sub>x<sub>23</sub> + dy<sub>21</sub>x<sub>32</sub> + dy<sub>22</sub>x<sub>33</sub> |

In TF, we can call `conv2d_backprop_filter` to get the dw.  As for the
computational pattern, it is still a convolution but with x as input and dy as
weight. Here is an example showing the results from `conv2d_backprop_filter` can
be matched by using `conv2d`.
```python
x = tf.reshape(tf.range(0, 9, dtype=tf.float32), (1, 3, 3, 1))
print("x:\n", x[0, :, :, 0].numpy())
dy = tf.ones((1, 2, 2, 1))
print("dy:\n", dy[0, :, :, 0].numpy())
dw = tf.compat.v1.nn.conv2d_backprop_filter(
    x, [2, 2, 1, 1], dy, [1, 1, 1, 1], 'VALID', use_cudnn_on_gpu=True,
    data_format='NHWC', dilations=[1, 1, 1, 1])
print("dw:\n", dw[:, :, 0, 0].numpy())
dy = tf.reshape(dy, (2, 2, 1, 1))
dw_copy = tf.nn.conv2d(x, dy, (1, 1), 'VALID', data_format='NHWC')
dw_copy = tf.reshape(dw_copy, (2, 2, 1, 1))
print("dw_equivalent:\n", dw_copy[:, :, 0, 0].numpy())
```
```
x:
 [[0. 1. 2.]
 [3. 4. 5.]
 [6. 7. 8.]]
dy:
 [[1. 1.]
 [1. 1.]]
dw:
 [[ 8. 12.]
 [20. 24.]]
dw_equivalent:
 [[ 8. 12.]
 [20. 24.]]
```

Similarly, the input gradients can be calculated by dx = ∂e/∂x = (∂e/∂y)(∂y/∂x) = dy⋅w.

Input gradient equations |
--- |
dx<sub>11</sub> = w<sub>11</sub>dy<sub>11</sub>                                                                                                 |
dx<sub>12</sub> = w<sub>12</sub>dy<sub>11</sub> + w<sub>11</sub>dy<sub>12</sub>                                                                 |
dx<sub>13</sub> = w<sub>12</sub>dy<sub>12</sub>                                                                                                 |
dx<sub>21</sub> = w<sub>21</sub>dy<sub>11</sub> + w<sub>11</sub>dy<sub>21</sub>                                                                 |
dx<sub>22</sub> = w<sub>22</sub>dy<sub>11</sub> + w<sub>21</sub>dy<sub>12</sub> + w<sub>12</sub>dy<sub>21</sub> + w<sub>11</sub>dy<sub>22</sub> |
dx<sub>23</sub> = w<sub>22</sub>dy<sub>12</sub> + w<sub>12</sub>dy<sub>22</sub>                                                                 |
dx<sub>31</sub> = w<sub>21</sub>dy<sub>21</sub>                                                                                                 |
dx<sub>32</sub> = w<sub>22</sub>dy<sub>21</sub> + w<sub>21</sub>dy<sub>22</sub>                                                                 |
dx<sub>33</sub> = w<sub>22</sub>dy<sub>22</sub>                                                                                                 |

In TF, we can call `conv2d_backprop_input` to get the dx. The computation
pattern is still a convolution but the input becomes the dy and the weight ends
up being a reversed w. So, to match the results from `conv2d_backprop_input`, we
need to conduct some padding over the dy and reverse the w before calling the
`conv2d`.
```python
dy = tf.ones((1, 2, 2, 1))
print("dy:\n", dy[0, :, :, 0].numpy())
w = tf.reshape(tf.range(0, 4, dtype=tf.float32), (2, 2, 1, 1))
print("w:\n", w[:, :, 0, 0].numpy())
dx = tf.compat.v1.nn.conv2d_backprop_input(
    (1, 3, 3, 1), filter=w, out_backprop=dy, strides=(1, 1, 1, 1),
    padding='VALID', use_cudnn_on_gpu=True, data_format='NHWC',
    dilations=[1, 1, 1, 1])
print("dx:\n", dx[0, :, :, 0].numpy())
dy = tf.pad(dy, [[0,0],[1,1],[1,1],[0,0]])
print("padded dy=\n", dy[0, :, :, 0].numpy())
w = tf.reverse(w, axis=[0, 1])
print("reversed w=\n", w[:, :, 0, 0].numpy())
dx_copy = tf.nn.conv2d(dy, w, (1, 1), 'VALID', data_format='NHWC')
print("dx_equivalent=\n", dx_copy[0, :, :, 0].numpy())
```
```
dy:
 [[1. 1.]
 [1. 1.]]
w:
 [[0. 1.]
 [2. 3.]]
dx:
 [[0. 1. 1.]
 [2. 6. 4.]
 [2. 5. 3.]]
padded dy=
 [[0. 0. 0. 0.]
 [0. 1. 1. 0.]
 [0. 1. 1. 0.]
 [0. 0. 0. 0.]]
reversed w=
 [[3. 2.]
 [1. 0.]]
dx_equivalent=
 [[0. 1. 1.]
 [2. 6. 4.]
 [2. 5. 3.]]
```


### Convolution in a Graph
Understanding the input/output of convolution forward/backward operations can
help us get an idea of how the graph is built when performing the training.

<p align=center> Fig 1. Convolution </p>
![Convolution In a Graph](/assets/posts_images/conv2d.PNG)

## BiasAdd Pattern
### BiasAdd Forward

BiasAdd equations |
--- |
y = x + b |

### BiasAdd Backward
db = ∂e/∂b = (∂e/∂y)(∂y/∂b) = dy
dx = ∂e/∂x = (∂e/∂y)(∂y/∂x) = dy

### BiasAdd in a Graph

<p align=center> Fig 2. BiasAdd </p>
![BiasAdd In a Graph](/assets/posts_images/bias.PNG)

## ReLU Pattern
### ReLU Forward

ReLU equations |
--- |
y = 0, x ≤ 0 |
y = x, x > 0 |

### ReLU Backward

Input gradient equations |
--- |
dx = 0, y ≤ 0 (or x ≤ 0) |
dx = dy, y > 0 (or x > 0) |

We use y rather than x, because it will be more friendly for fusion.
### ReLU in a Graph

<p align=center> Fig 3. ReLU </p>
![ReLU In a Graph](/assets/posts_images/relu.PNG)

## Putting Them All Together

<p align=center> Fig 4. Fused Ops </p>
![All In a Graph](/assets/posts_images/fuse.PNG)
