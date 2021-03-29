---
layout: posts
title:  "Demystifying the BatchNorm-Add-ReLU Fusion"
published: true
author: kaixi_hou
search                   : true
search_full_content      : true
search_provider          : google
comments: true
---
## Introduction
My previous post, "[Demystifying the Conv-Bias-ReLU
Fusion](https://kaixih.github.io/cbr-fusion/)", has introduced a common fusion
pattern in deep learning models. This post, on the other hand, will discuss
another fusion pattern BatchNorm-Add-ReLU that also can be found in many
models, such as ResNet50. Unlike the previous post, we will investigate the
feasibility of the fusion for both forward and backprop stages.

## BatchNorm Pattern
As the convolution in the Conv-Bias-ReLU pattern, the BatchNorm is the most
significant node in the BatchNorm-Add-ReLU pattern. Many articles have already
demonstrated how the batch norm works and its backpropagation derived such as [this
one](https://kevinzakka.github.io/2016/09/14/batch_normalization/). For
simplicity, here we only need to know what the inputs and outputs of the batch
norm in its forward and backward passes.
* Forward pass: it basically requires an x as the input and gamma/beta (γ/β) as
  two trainable variables. It can then output y.
* Backward pass: it requires the backpropagated gradient input dy as well as the
  x and γ from the forward op to output the dx, dγ, and dβ. Note the β is
  not needed.

## Add Pattern
We assume the add op is a simple binary addition and thus we need two input x
and z (also called the "side input") and then output y. The backprop is to get dx
and dz by using the backpropagated gradient input dy. As the BiasAdd shown in
"[this page](https://kaixih.github.io/cbr-fusion/)", the backprop is simply an
"Identity" op to forward the dy and it doesn't require any input from the
forward pass. The equations are below:

Add equations (forward) |
--- |
y = x + z |

Add equations (backward) |
--- |
dx = ∂e/∂x = (∂e/∂y)(∂y/∂x) = dy |
dz = ∂e/∂z = (∂e/∂y)(∂y/∂z) = dy |


## ReLU Pattern
The forward Relu is quite straightforward that we only need one input x and one
output y.  In contrast, to compute the dx, the backward Relu can either rely on
x or y to pass the given backpropagated gradient dy. Mathematically, they are
same but using y would be more "fusion-friendly", since the x will become the
"intermediate results" and be hard to access if the fusion is applied on
BatchNorm-Add-ReLU.

ReLU equations (forward) |
--- |
y = 0, x ≤ 0 |
y = x, x > 0 |

ReLU equations (backward) |
--- |
dx = 0, y ≤ 0 (or x ≤ 0) |
dx = dy, y > 0 (or x > 0) |

## Putting Them All Together
Now, we can draw a figure to show how we can fuse these three operations. Based
on the above analysis, the BatchNorm-Add-ReLU can be safely fused into one
operation since the backward pass won't use any intemediate results from the
fused operation.  The fused forward op will need input x, gamma/beta and side
input z, and finally output y. For the backward pass, the ReluGrad and
BatchNormGrad can also be fused together, which requires the backpropagated
gradient dy and the output y, the input x and input gamma from the forward op to output
the dx (input gradient), dγ/dβ (varialbel graidents), and dz (side input
gradient).

<p align=center> Fig 4. Fused Ops for BatchNorm+Add+ReLU </p> ![All
In a Graph](/assets/posts_images/bn_act_fuse.png)

It is still worth to mention that this post focuses mainly on the scenario of training
and discusses the fusion from the perspective of the data dependencies. In
reality, the decision to fuse will be more complex than it
seems.

