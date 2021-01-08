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

<p style="font-size:11px">
<table>
  <tr>
    <th>Month</th>
    <th>Savings</th>
  </tr>
  <tr>
    <td><span style="background-color: green">January</span></td>
    <td>$100</td>
  </tr>
  <tr>
    <td>February</td>
    <td>$80</td>
  </tr>
</table>

|<span style="background-color: green"> 0.01492912 </span>|<span style="background-color: green">-0.08340913 </span>|<span style="background-color: green">-0.13510555 </span>|<span style="background-color: red">0.7274594   </span>|<span style="background-color: red">0.27867514 </span>|<span style="background-color: red">-0.22769517 </span>|<span style="background-color: yellow">-0.09443533 </span>|<span style="background-color: yellow">0.14927697 </span>|<span style="background-color: yellow">-0.06407022 </span>|
|<span style="background-color: green"> 0.3732596  </span>|<span style="background-color: green">-0.46085864 </span>|<span style="background-color: green"> 0.0720188  </span>|<span style="background-color: red">0.07225263  </span>|<span style="background-color: red">0.0731563  </span>|<span style="background-color: red">-0.32511705 </span>|<span style="background-color: yellow">-0.5776096  </span>|<span style="background-color: yellow">0.19336885 </span>|<span style="background-color: yellow"> 0.5521658  </span>|

</p>

Keras Recurrent Weights: 

|<span style="background-color: #8FBC8F">-0.1763835  </span>|<span style="background-color: #8FBC8F">-0.34464398 </span>|<span style="background-color: #8FBC8F">-0.68863446 </span>|<span style="background-color: #CD5C5C">-0.26089588 </span>|<span style="background-color: #CD5C5C">-0.07611549 </span>|<span style="background-color: #CD5C5C">-0.3227275 </span>|<span style="background-color: #FFFFE0">0.27895766 </span>|<span style="background-color: #FFFFE0"> 0.00449633 </span>|<span style="background-color: #FFFFE0"> 0.34646943</span>|
|<span style="background-color: #8FBC8F">-0.20453213 </span>|<span style="background-color: #8FBC8F"> 0.10408226 </span>|<span style="background-color: #8FBC8F">-0.31350893 </span>|<span style="background-color: #CD5C5C"> 0.49217793 </span>|<span style="background-color: #CD5C5C"> 0.2363063  </span>|<span style="background-color: #CD5C5C"> 0.1172057 </span>|<span style="background-color: #FFFFE0">0.5199497  </span>|<span style="background-color: #FFFFE0">-0.08515472 </span>|<span style="background-color: #FFFFE0">-0.50953877</span>|
|<span style="background-color: #8FBC8F"> 0.30824518 </span>|<span style="background-color: #8FBC8F"> 0.0503802  </span>|<span style="background-color: #8FBC8F">-0.25397384 </span>|<span style="background-color: #CD5C5C">-0.538845   </span>|<span style="background-color: #CD5C5C"> 0.24127854 </span>|<span style="background-color: #CD5C5C"> 0.4379759 </span>|<span style="background-color: #FFFFE0">-0.03005415</span>|<span style="background-color: #FFFFE0">-0.50177294 </span>|<span style="background-color: #FFFFE0">-0.2118314 </span>|

Keras Biases: 
[[-0.02635479 -0.02612267  0.00036312  0.02735401  0.01107747  0.03721783 -0.02271527  0.01183162 -0.02974842]
 [ 0.03700757 -0.00075941 -0.00030701 -0.04698812  0.01857647  0.01315722 -0.02921613 -0.00608767 -0.03110452]]

Weights:

|<span style="background-color: red"> 0.727459 </span>|<span style="background-color: red"> 0.072253 </span>|<span style="background-color: red"> 0.278675 </span>|<span style="background-color: red"> 0.073156 </span>|<span style="background-color: red">-0.227695 </span>|<span style="background-color: red">-0.325117 </span>|<span style="background-color: green"> 0.014929 </span>|<span style="background-color: green"> 0.373260 </span>|<span style="background-color: green">-0.083409 </span>|<span style="background-color: green">-0.460859</span>|
|<span style="background-color: green">-0.135106 </span>| <span style="background-color: green">0.072019 </span>|<span style="background-color: yellow">-0.094435 </span>|<span style="background-color: yellow">-0.577610 </span>|<span style="background-color: yellow"> 0.149277 </span>|<span style="background-color: yellow"> 0.193369 </span>|<span style="background-color: yellow">-0.064070 </span>|<span style="background-color: yellow"> 0.552166 </span>|<span style="background-color: #CD5C5C">-0.260896 </span>|<span style="background-color: #CD5C5C"> 0.492178</span>|
|<span style="background-color: #CD5C5C">-0.538845 </span>|<span style="background-color: #CD5C5C">-0.076115 </span>|<span style="background-color: #CD5C5C"> 0.236306 </span>|<span style="background-color: #CD5C5C"> 0.241279 </span>|<span style="background-color: #CD5C5C">-0.322728 </span>|<span style="background-color: #CD5C5C"> 0.117206   </span>|<span style="background-color: #CD5C5C"> 0.437976 </span>|<span style="background-color: #8FBC8F">-0.176383 </span>|<span style="background-color: #8FBC8F">-0.204532 </span>|<span style="background-color: #8FBC8F"> 0.308245</span>|
|<span style="background-color: #8FBC8F">-0.344644 </span>|<span style="background-color: #8FBC8F"> 0.104082 </span>|<span style="background-color: #8FBC8F"> 0.050380 </span>|<span style="background-color: #8FBC8F">-0.688634 </span>|<span style="background-color: #8FBC8F">-0.313509 </span>| <span style="background-color: #8FBC8F">-0.253974 </span>|<span style="background-color: #FFFFE0"> 0.278958 </span>|<span style="background-color: #FFFFE0"> 0.519950 </span>|<span style="background-color: #FFFFE0">-0.030054 </span>|<span style="background-color: #FFFFE0"> 0.004496</span>|
|<span style="background-color: #FFFFE0">-0.085155 </span>|<span style="background-color: #FFFFE0">-0.501773 </span>|<span style="background-color: #FFFFE0"> 0.346469 </span>|<span style="background-color: #FFFFE0">-0.509539 </span>|<span style="background-color: #FFFFE0">-0.211831 </span>| <span style="background-color: "> 0.027354 </span>|<span style="background-color: "> 0.011077 </span>|<span style="background-color: "> 0.037218 </span>|<span style="background-color: ">-0.026355 </span>|<span style="background-color: ">-0.026123</span>|
|<span style="background-color: "> 0.000363 </span>|<span style="background-color: ">-0.022715 </span>|<span style="background-color: "> 0.011832 </span>|<span style="background-color: ">-0.029748 </span>|<span style="background-color: ">-0.046988 </span>| <span style="background-color: "> 0.018576 </span>|<span style="background-color: "> 0.013157 </span>|<span style="background-color: "> 0.037008 </span>|<span style="background-color: ">-0.000759 </span>|<span style="background-color: ">-0.000307</span>|
|<span style="background-color: ">-0.029216 </span>|<span style="background-color: ">-0.006088 </span>|<span style="background-color: ">-0.031105 </span>|

(1) the array for the cudnn is a flat array which consists of all kernels and biases.
(2) The general order is sill kernel wieights , recurrent weights and biases, However, the Wi and Wr's order is swapped, meaning the corresponding biases also need to swap.

## LSTM

## Reference
* [NVIDIA CUDNN Documentation: cudnnRNNMode_t](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNMode_t)

