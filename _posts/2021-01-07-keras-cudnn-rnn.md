---
layout: posts
title:  "Tensorflow Keras RNN and its Backend CUDNN RNN"
author: kaixi_hou
#search                   : true
#search_full_content      : true
#search_provider          : google
comments: true
---
## Introduction
For RNN, people might want to port the code from keras to cudnn, or vice versa,
or different use one of them as reference for their implementation. The keras RNN
uses the cudnn as the backend, however, it isn't directly call it under the hood.
They usually need to preprocess the inputs and then call the library, one facter
is the layout of the weights and biases used in Keras and CUDNN is different,
which might cause confusion for many develepers.

This post will check the popular GRU and LSTM layer from Keras and CUDNN and discuss
the some of the details of how the weights/biases are organized for them and how
to transform your parameters from one API to another and the slight difference
of the formula used in these two API.

Note: this post assumes people has already understand the RNN structure of GRU
and LSTM. And will borrow the formula from NVIDIA CUDNN documentation for reference.

## GRU
it = σ(Wixt + Riht-1 + bWi + bRi)
rt = σ(Wrxt + Rrht-1 + bWr + bRr)
h't = tanh(Whxt + rt◦(Rhht-1 + bRh) + bWh)
ht = (1 - it) ◦ h't + it ◦ ht-1
hiddenSize  = 3
inputSize  = 2
This formula represents a double bias senario, meaning for each weights by input multiplication, we will apply a bias addition.
We need 3 kernel weights Wi Wr Wh and 3 recurrent weights (Ri Rr Rh), also 3 biases for each.
Wi should by inputSize by hiddenSize. Ri will be hiddenSize by hiddenSize. Bias will be same with hiddenSize.
Suppose we have GRU layer in keras, the kernel/recurrent weights will be laid out as inputSize by 3(hiddenSize) while the recurent will be hiddenSize by 3(hiddenSize).
For example, the following code is used to put out the weights and bias stored in Keras:
And it also shows the configuration of GRU layer.

This following shows under the hood what the CUDNN is receiving for the weights array and we can notice two things:

Keras Kernel Weights: 


|<span style="background-color: green"> 0.01492912 </span>|-0.08340913 |-0.13510555  |0.7274594   |0.27867514 |-0.22769517 |-0.09443533  |0.14927697 |-0.06407022 |
|<span style="background-color: green"> 0.3732596  </span>|-0.46085864 | 0.0720188   |0.07225263  |0.0731563  |-0.32511705 |-0.5776096   |0.19336885 | 0.5521658  |

[[ 0.01492912 -0.08340913 -0.13510555  0.7274594   0.27867514 -0.22769517 -0.09443533  0.14927697 -0.06407022]
 [ 0.3732596  -0.46085864  0.0720188   0.07225263  0.0731563  -0.32511705 -0.5776096   0.19336885  0.5521658 ]]
Keras Recurrent Weights: 
[[-0.1763835  -0.34464398 -0.68863446 -0.26089588 -0.07611549 -0.3227275 0.27895766  0.00449633  0.34646943]
 [-0.20453213  0.10408226 -0.31350893  0.49217793  0.2363063   0.1172057 0.5199497  -0.08515472 -0.50953877]
 [ 0.30824518  0.0503802  -0.25397384 -0.538845    0.24127854  0.4379759 -0.03005415 -0.50177294 -0.2118314 ]]
Keras Biases: 
[[-0.02635479 -0.02612267  0.00036312  0.02735401  0.01107747  0.03721783 -0.02271527  0.01183162 -0.02974842]
 [ 0.03700757 -0.00075941 -0.00030701 -0.04698812  0.01857647  0.01315722 -0.02921613 -0.00608767 -0.03110452]]

Weights:
0.727459 0.072253 0.278675 0.073156 -0.227695 -0.325117 0.014929 0.373260 -0.083409 -0.460859
-0.135106 0.072019 -0.094435 -0.577610 0.149277 0.193369 -0.064070 0.552166 -0.260896 0.492178
-0.538845 -0.076115 0.236306 0.241279 -0.322728 0.117206 0.437976 -0.176383 -0.204532 0.308245
-0.344644 0.104082 0.050380 -0.688634 -0.313509 -0.253974 0.278958 0.519950 -0.030054 0.004496
-0.085155 -0.501773 0.346469 -0.509539 -0.211831 0.027354 0.011077 0.037218 -0.026355 -0.026123
0.000363 -0.022715 0.011832 -0.029748 -0.046988 0.018576 0.013157 0.037008 -0.000759 -0.000307
-0.029216 -0.006088 -0.031105

(1) the array for the cudnn is a flat array which consists of all kernels and biases.
(2) The general order is sill kernel wieights , recurrent weights and biases, However, the Wi and Wr's order is swapped, meaning the corresponding biases also need to swap.

## LSTM

## Reference
* [NVIDIA CUDNN Documentation: cudnnRNNMode_t](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNMode_t)

