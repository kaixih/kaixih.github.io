---
layout: posts
title:  "Demystifying the Conv-Bias-ReLU Fusion"
published: true
author: kaixi_hou
search                   : true
search_full_content      : true
search_provider          : google
comments: true
---
## Introduction
My previous post, "[Fused Operations in
Tensorflow](https://kaixih.github.io/fused-api/)", introduced the basics of
operation fusion in deep learning by showing how to enable the grappler
optimizer in Tensorflow to recognize the supported patterns and then fuse them
together for better performance. In that post, I briefly talked about the
Conv-Bias-Relu pattern, which is a great fit for fusion. In this post, by
constrast, I will dive deeper into the Conv-Bias-Relu computation pattern and
discuss why and how it can be fused.

## Convolution Pattern
Let's start from the convolution shown in the following figure, which takes two
parameters - a 3x3 input and a 2x2 weight - and outputs a 2x2 array.

<p align=center> Fig 0. Convolution's Computational Pattern </p>
![Convolution Pattern](/assets/posts_images/conv_pattern.PNG)

### Convolution Forward Pass
The convolution forward pass computes a weighted sum of the current input
element as well as its surrounding neighbors. The process can be much easier to
understand with the equations shown as below that matches the above Fig.0.

Convolution equations |
--- |
y<sub>11</sub> = w<sub>11</sub>x<sub>11</sub> + w<sub>12</sub>x<sub>12</sub> + w<sub>21</sub>x<sub>21</sub> + w<sub>22</sub>x<sub>22</sub> |
y<sub>12</sub> = w<sub>11</sub>x<sub>12</sub> + w<sub>12</sub>x<sub>13</sub> + w<sub>21</sub>x<sub>22</sub> + w<sub>22</sub>x<sub>23</sub> |
y<sub>21</sub> = w<sub>11</sub>x<sub>21</sub> + w<sub>12</sub>x<sub>22</sub> + w<sub>21</sub>x<sub>31</sub> + w<sub>22</sub>x<sub>32</sub> |
y<sub>22</sub> = w<sub>11</sub>x<sub>22</sub> + w<sub>12</sub>x<sub>23</sub> + w<sub>21</sub>x<sub>32</sub> + w<sub>22</sub>x<sub>33</sub> |

Here w, x, and y are weight, input, and output arrays respectively. To get a
better sense of how the Tensorflow API does this, let's have a look at a code
snippet of using `tf.nn.conv2d` to perform above computation. In the example, we
use the synthetic data for the x and w.
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

### Convolution Backward Pass
The convolution backward pass is to compute the gradients of w and x. Let's
suppose e is the error returned by any cost/loss function and thus the gradients
of x and w are written as dw (= ∂e/∂w) and dx (= ∂e/∂x). According to the chain
rule, we can easily get dw = ∂e/∂w = (∂e/∂y)(∂y/∂w) = dy⋅x. More precisely, the
equations for dw are:

Weight gradient equations |
--- |
dw<sub>11</sub> = dy<sub>11</sub>x<sub>11</sub> + dy<sub>12</sub>x<sub>12</sub> + dy<sub>21</sub>x<sub>21</sub> + dy<sub>22</sub>x<sub>22</sub> |
dw<sub>12</sub> = dy<sub>11</sub>x<sub>12</sub> + dy<sub>12</sub>x<sub>13</sub> + dy<sub>21</sub>x<sub>22</sub> + dy<sub>22</sub>x<sub>23</sub> |
dw<sub>21</sub> = dy<sub>11</sub>x<sub>21</sub> + dy<sub>12</sub>x<sub>22</sub> + dy<sub>21</sub>x<sub>31</sub> + dy<sub>22</sub>x<sub>32</sub> |
dw<sub>22</sub> = dy<sub>11</sub>x<sub>22</sub> + dy<sub>12</sub>x<sub>23</sub> + dy<sub>21</sub>x<sub>32</sub> + dy<sub>22</sub>x<sub>33</sub> |

In Tensorflow, `tf.compat.v1.nn.conv2d_backprop_filter` is used to calculate the
dw. It should be noted that though `conv2d_backprop_filter` is a separate API,
its computation pattern is essentially a convolutin but with the x as the input array
and dy as the weight array. Therefore, for learning purposes we can still call `conv2d` to realize its
functionality. The following script shows the results from
`conv2d_backprop_filter` can be matched with `conv2d`. In the test, the x is
synthetic data and we assume the dy is full of ones.
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

Similarly, the input gradients can be calculated by dx = ∂e/∂x = (∂e/∂y)(∂y/∂x)
= dy⋅w. From the equations below, the computation pattern is actually still a
convolution but the input and weight end up being the dy and a reversed w. 

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

In Tensorflow, we have `tf.compat.v1.nn.conv2d_backprop_input` to compute the
dx. In addition, to match its results, we can still use `conv2d` but need to pad
the dy and reverse the w before the call.  The script shows this process with
synthetic data in w and all ones in dy.
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
If we put all the input/output tensors and operation nodes into one graph, we
can see the data flow and dependencies more clearly. The takeaway here is that
the input x and w for the forward pass is still needed in backward convolution
to compute dw and dx respectively. In other words, both the input x and w need
to be alive even when the forward pass has already done. Whereas, the output y
from the forward convolution will no longer be used in backward pass.

<p align=center> Fig 1. Convolution </p>
![Convolution In a Graph](/assets/posts_images/conv2d.PNG)

## BiasAdd Pattern
### BiasAdd Forward Pass
Compared to the convolution, the bias add is much simpler. The following
equation shows that we add the input x with the bias b to obtain y.

BiasAdd equations |
--- |
y = x + b |

### BiasAdd Backward Pass
Since the bias b is a trainable parameter, we use the following equations to get
the db as well as dx, which are essentially a forward operation of dy.

Bias/Input gradient equations |
--- |
db = ∂e/∂b = (∂e/∂y)(∂y/∂b) = dy |
dx = ∂e/∂x = (∂e/∂y)(∂y/∂x) = dy |

### BiasAdd in a Graph
The figure below shows the bias add operations. Apparently, neither of the input
nor the output from the forward pass is needed in the backward pass.

<p align=center> Fig 2. BiasAdd </p>
![BiasAdd In a Graph](/assets/posts_images/bias.PNG)

## ReLU Pattern
### ReLU Forward Pass
The ReLU is also straightforward. From the equation below, we can learn that
there is no trainable parameters and we only have one input x and one output y.

ReLU equations |
--- |
y = 0, x ≤ 0 |
y = x, x > 0 |

### ReLU Backward Pass
The backward pass only need to compute the dx, and to do so we can use x or y.
Mathematically, they are same but using y would be more "fusion-friendly", which
will be explained later.

Input gradient equations |
--- |
dx = 0, y ≤ 0 (or x ≤ 0) |
dx = dy, y > 0 (or x > 0) |

### ReLU in a Graph
After we put all nodes in a graph, we can observe the backward pass only needs
the output from the forward pass.

<p align=center> Fig 3. ReLU </p>
![ReLU In a Graph](/assets/posts_images/relu.PNG)

## Putting Them All Together
Now, we can draw all these three operations together in one figure. Based on the
above analysis, the Conv-Bias-Relu can be safely fused into one operation since
the backward pass won't use any intemediate results from the fused operation but
only its input x, w and b and its output y. 
<p align=center> Fig 4. Fused Ops </p>
![All In a Graph](/assets/posts_images/fuse.PNG)

It is worth to mention that this post focuses mainly on the scenario of training
and discusses the fusion from the perspective of the data dependencies. In
reality, the decision to fuse will be more complex than it
seems.
