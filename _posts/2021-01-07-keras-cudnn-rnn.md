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
<table border=0px style="font-size:13px;">
  <tr>
    <td style="background-color: green">  0.014929</td>
    <td style="background-color: green"> -0.083409</td>
    <td style="background-color: green"> -0.135106</td>
    <td style="background-color: red">    0.727459</td>
    <td style="background-color: red">    0.278675</td>
    <td style="background-color: red">   -0.227695</td>
    <td style="background-color: yellow">-0.094435</td>
    <td style="background-color: yellow"> 0.149277</td>
    <td style="background-color: yellow">-0.064070</td>
  </tr>
  <tr>
    <td style="background-color: green">  0.373260</td>
    <td style="background-color: green"> -0.460859</td>
    <td style="background-color: green">  0.072019</td>
    <td style="background-color: red">    0.072253</td>
    <td style="background-color: red">    0.073156</td>
    <td style="background-color: red">   -0.325117</td>
    <td style="background-color: yellow">-0.577610</td>
    <td style="background-color: yellow"> 0.193369</td>
    <td style="background-color: yellow"> 0.552166</td>
  </tr>
</table>


Keras Recurrent Weights: 
<table border=0px style="font-size:13px;">
  <tr>
    <td style="background-color: #8FBC8F"> -0.176383</td>
    <td style="background-color: #8FBC8F"> -0.344644</td>
    <td style="background-color: #8FBC8F"> -0.688634</td>
    <td style="background-color: #CD5C5C"> -0.260896</td>
    <td style="background-color: #CD5C5C"> -0.076115</td>
    <td style="background-color: #CD5C5C"> -0.322728</td>
    <td style="background-color: #FFFFE0">  0.278958</td>
    <td style="background-color: #FFFFE0">  0.004496</td>
    <td style="background-color: #FFFFE0">  0.346469</td>
  </tr>
  <tr>
    <td style="background-color: #8FBC8F"> -0.204532</td>
    <td style="background-color: #8FBC8F">  0.104082</td>
    <td style="background-color: #8FBC8F"> -0.313509</td>
    <td style="background-color: #CD5C5C">  0.492178</td>
    <td style="background-color: #CD5C5C">  0.236306</td>
    <td style="background-color: #CD5C5C">  0.117206</td>
    <td style="background-color: #FFFFE0">  0.519950</td>
    <td style="background-color: #FFFFE0"> -0.085155</td>
    <td style="background-color: #FFFFE0"> -0.509539</td>
  </tr>
  <tr>
    <td style="background-color: #8FBC8F">  0.308245</td>
    <td style="background-color: #8FBC8F">  0.050380</td>
    <td style="background-color: #8FBC8F"> -0.253974</td>
    <td style="background-color: #CD5C5C"> -0.538845</td>
    <td style="background-color: #CD5C5C">  0.241279</td>
    <td style="background-color: #CD5C5C">  0.437976</td>
    <td style="background-color: #FFFFE0"> -0.030054</td>
    <td style="background-color: #FFFFE0"> -0.501773</td>
    <td style="background-color: #FFFFE0"> -0.211831</td>
  </tr>
</table>

Keras Biases: 
<table border=0px style="font-size:13px;">
  <tr>
    <td style="background-color: #006400"> -0.026355</td>
    <td style="background-color: #006400"> -0.026123</td>
    <td style="background-color: #006400">  0.000363</td>
    <td style="background-color: #8B0000">  0.027354</td>
    <td style="background-color: #8B0000">  0.011077</td>
    <td style="background-color: #8B0000">  0.037218</td>
    <td style="background-color: #FFD700"> -0.022715</td>
    <td style="background-color: #FFD700">  0.011832</td>
    <td style="background-color: #FFD700"> -0.029748</td>
  </tr>
  <tr>
    <td style="background-color: #006400">  0.037008</td>
    <td style="background-color: #006400"> -0.000759</td>
    <td style="background-color: #006400"> -0.000307</td>
    <td style="background-color: #8B0000"> -0.046988</td>
    <td style="background-color: #8B0000">  0.018576</td>
    <td style="background-color: #8B0000">  0.013157</td>
    <td style="background-color: #FFD700"> -0.029216</td>
    <td style="background-color: #FFD700"> -0.006088</td>
    <td style="background-color: #FFD700"> -0.031105</td>
  </tr>
</table>

CUDNN Weights:
<table border=0px style="font-size:13px;">
  <tr>
    <td style="background-color: red">    0.727459</td>
    <td style="background-color: red">    0.072253</td>
    <td style="background-color: red">    0.278675</td>
    <td style="background-color: red">    0.073156</td>
    <td style="background-color: red">   -0.227695</td>
    <td style="background-color: red">   -0.325117</td>
    <td style="background-color: green">  0.014929</td>
    <td style="background-color: green">  0.373260</td>
    <td style="background-color: green"> -0.083409</td>
    <td style="background-color: green"> -0.460859</td>
  </tr>
  <tr>
    <td style="background-color: green">   -0.135106</td>
    <td style="background-color: green">    0.072019</td>
    <td style="background-color: yellow">  -0.094435</td>
    <td style="background-color: yellow">  -0.577610</td>
    <td style="background-color: yellow">   0.149277</td>
    <td style="background-color: yellow">   0.193369</td>
    <td style="background-color: yellow">  -0.064070</td>
    <td style="background-color: yellow">   0.552166</td>
    <td style="background-color: #CD5C5C"> -0.260896</td>
    <td style="background-color: #CD5C5C">  0.492178</td>
  </tr>
  <tr>
    <td style="background-color: #CD5C5C"> -0.538845</td>
    <td style="background-color: #CD5C5C"> -0.076115</td>
    <td style="background-color: #CD5C5C">  0.236306</td>
    <td style="background-color: #CD5C5C">  0.241279</td>
    <td style="background-color: #CD5C5C"> -0.322728</td>
    <td style="background-color: #CD5C5C">  0.117206</td>
    <td style="background-color: #CD5C5C">  0.437976</td>
    <td style="background-color: #8FBC8F"> -0.176383</td>
    <td style="background-color: #8FBC8F"> -0.204532</td>
    <td style="background-color: #8FBC8F">  0.308245</td>
  </tr>
  <tr>
    <td style="background-color: #8FBC8F"> -0.344644</td>
    <td style="background-color: #8FBC8F">  0.104082</td>
    <td style="background-color: #8FBC8F">  0.050380</td>
    <td style="background-color: #8FBC8F"> -0.688634</td>
    <td style="background-color: #8FBC8F"> -0.313509</td>
    <td style="background-color: #8FBC8F"> -0.253974</td>
    <td style="background-color: #FFFFE0">  0.278958</td>
    <td style="background-color: #FFFFE0">  0.519950</td>
    <td style="background-color: #FFFFE0"> -0.030054</td>
    <td style="background-color: #FFFFE0">  0.004496</td>
  </tr>
  <tr>
    <td style="background-color: #FFFFE0"> -0.085155</td>
    <td style="background-color: #FFFFE0"> -0.501773</td>
    <td style="background-color: #FFFFE0">  0.346469</td>
    <td style="background-color: #FFFFE0"> -0.509539</td>
    <td style="background-color: #FFFFE0"> -0.211831</td>
    <td style="background-color: #8B0000">  0.027354</td>
    <td style="background-color: #8B0000">  0.011077</td>
    <td style="background-color: #8B0000">  0.037218</td>
    <td style="background-color: #006400"> -0.026355</td>
    <td style="background-color: #006400"> -0.026123</td>
  </tr>
  <tr>
    <td style="background-color: #006400">  0.000363</td>
    <td style="background-color: #FFD700"> -0.022715</td>
    <td style="background-color: #FFD700">  0.011832</td>
    <td style="background-color: #FFD700"> -0.029748</td>
    <td style="background-color: #8B0000"> -0.046988</td>
    <td style="background-color: #8B0000">  0.018576</td>
    <td style="background-color: #8B0000">  0.013157</td>
    <td style="background-color: #006400">  0.037008</td>
    <td style="background-color: #006400"> -0.000759</td>
    <td style="background-color: #006400"> -0.000307</td>
  </tr>
  <tr>
    <td style="background-color: #FFD700"> -0.029216</td>
    <td style="background-color: #FFD700"> -0.006088</td>
    <td style="background-color: #FFD700"> -0.031105</td>
  </tr>
</table>


(1) the array for the cudnn is a flat array which consists of all kernels and biases.
(2) The general order is sill kernel wieights , recurrent weights and biases, However, the Wi and Wr's order is swapped, meaning the corresponding biases also need to swap.

## LSTM

## Reference
* [NVIDIA CUDNN Documentation: cudnnRNNMode_t](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNMode_t)

