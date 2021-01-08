---
layout: posts
title:  "Tensorflow Keras RNN and its Backend CUDNN RNN"
author: kaixi_hou
#search                   : true
#search_full_content      : true
#search_provider          : google
comments: true
---
(DRAFT)
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
<!---
#90EE90 = lightgreen 
#CD5C5C = indianred
#FFFFE0 = lightyellow
#008000 = green
#FF0000 = red
#FFFF00 = yellow
#006400 = darkgreen
#8B0000 = darkred
#FFD700 = gold --->
<table border=0px style="font-size:13px;">
  <tr>
    <td style="background-color: #90EE90">  0.014929</td>
    <td style="background-color: #90EE90"> -0.083409</td>
    <td style="background-color: #90EE90"> -0.135106</td>
    <td style="background-color: #CD5C5C">  0.727459</td>
    <td style="background-color: #CD5C5C">  0.278675</td>
    <td style="background-color: #CD5C5C"> -0.227695</td>
    <td style="background-color: #FFFFE0"> -0.094435</td>
    <td style="background-color: #FFFFE0">  0.149277</td>
    <td style="background-color: #FFFFE0"> -0.064070</td>
  </tr>
  <tr>
    <td style="background-color: #90EE90">  0.373260</td>
    <td style="background-color: #90EE90"> -0.460859</td>
    <td style="background-color: #90EE90">  0.072019</td>
    <td style="background-color: #CD5C5C">  0.072253</td>
    <td style="background-color: #CD5C5C">  0.073156</td>
    <td style="background-color: #CD5C5C"> -0.325117</td>
    <td style="background-color: #FFFFE0"> -0.577610</td>
    <td style="background-color: #FFFFE0">  0.193369</td>
    <td style="background-color: #FFFFE0">  0.552166</td>
  </tr>
</table>


Keras Recurrent Weights: 
<table border=0px style="font-size:13px;">
  <tr>
    <td style="background-color: #008000"> -0.176383</td>
    <td style="background-color: #008000"> -0.344644</td>
    <td style="background-color: #008000"> -0.688634</td>
    <td style="background-color: #FF0000"> -0.260896</td>
    <td style="background-color: #FF0000"> -0.076115</td>
    <td style="background-color: #FF0000"> -0.322728</td>
    <td style="background-color: #FFFF00">  0.278958</td>
    <td style="background-color: #FFFF00">  0.004496</td>
    <td style="background-color: #FFFF00">  0.346469</td>
  </tr>
  <tr>
    <td style="background-color: #008000"> -0.204532</td>
    <td style="background-color: #008000">  0.104082</td>
    <td style="background-color: #008000"> -0.313509</td>
    <td style="background-color: #FF0000">  0.492178</td>
    <td style="background-color: #FF0000">  0.236306</td>
    <td style="background-color: #FF0000">  0.117206</td>
    <td style="background-color: #FFFF00">  0.519950</td>
    <td style="background-color: #FFFF00"> -0.085155</td>
    <td style="background-color: #FFFF00"> -0.509539</td>
  </tr>
  <tr>
    <td style="background-color: #008000">  0.308245</td>
    <td style="background-color: #008000">  0.050380</td>
    <td style="background-color: #008000"> -0.253974</td>
    <td style="background-color: #FF0000"> -0.538845</td>
    <td style="background-color: #FF0000">  0.241279</td>
    <td style="background-color: #FF0000">  0.437976</td>
    <td style="background-color: #FFFF00"> -0.030054</td>
    <td style="background-color: #FFFF00"> -0.501773</td>
    <td style="background-color: #FFFF00"> -0.211831</td>
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
    <td style="background-color: #CD5C5C">  0.727459</td>
    <td style="background-color: #CD5C5C">  0.072253</td>
    <td style="background-color: #CD5C5C">  0.278675</td>
    <td style="background-color: #CD5C5C">  0.073156</td>
    <td style="background-color: #CD5C5C"> -0.227695</td>
    <td style="background-color: #CD5C5C"> -0.325117</td>
    <td style="background-color: #90EE90">  0.014929</td>
    <td style="background-color: #90EE90">  0.373260</td>
    <td style="background-color: #90EE90"> -0.083409</td>
  </tr>
  <tr>
    <td style="background-color: #90EE90"> -0.460859</td>
    <td style="background-color: #90EE90"> -0.135106</td>
    <td style="background-color: #90EE90">  0.072019</td>
    <td style="background-color: #FFFFE0"> -0.094435</td>
    <td style="background-color: #FFFFE0"> -0.577610</td>
    <td style="background-color: #FFFFE0">  0.149277</td>
    <td style="background-color: #FFFFE0">  0.193369</td>
    <td style="background-color: #FFFFE0"> -0.064070</td>
    <td style="background-color: #FFFFE0">  0.552166</td>
  </tr>
  <tr>
    <td style="background-color: #FF0000"> -0.260896</td>
    <td style="background-color: #FF0000">  0.492178</td>
    <td style="background-color: #FF0000"> -0.538845</td>
    <td style="background-color: #FF0000"> -0.076115</td>
    <td style="background-color: #FF0000">  0.236306</td>
    <td style="background-color: #FF0000">  0.241279</td>
    <td style="background-color: #FF0000"> -0.322728</td>
    <td style="background-color: #FF0000">  0.117206</td>
    <td style="background-color: #FF0000">  0.437976</td>
  </tr>
  <tr>
    <td style="background-color: #008000"> -0.176383</td>
    <td style="background-color: #008000"> -0.204532</td>
    <td style="background-color: #008000">  0.308245</td>
    <td style="background-color: #008000"> -0.344644</td>
    <td style="background-color: #008000">  0.104082</td>
    <td style="background-color: #008000">  0.050380</td>
    <td style="background-color: #008000"> -0.688634</td>
    <td style="background-color: #008000"> -0.313509</td>
    <td style="background-color: #008000"> -0.253974</td>
  </tr>
  <tr>
    <td style="background-color: #FFFFE0">  0.278958</td>
    <td style="background-color: #FFFFE0">  0.519950</td>
    <td style="background-color: #FFFFE0"> -0.030054</td>
    <td style="background-color: #FFFFE0">  0.004496</td>
    <td style="background-color: #FFFFE0"> -0.085155</td>
    <td style="background-color: #FFFFE0"> -0.501773</td>
    <td style="background-color: #FFFFE0">  0.346469</td>
    <td style="background-color: #FFFFE0"> -0.509539</td>
    <td style="background-color: #FFFFE0"> -0.211831</td>
  </tr>
  <tr>
    <td style="background-color: #8B0000">  0.027354</td>
    <td style="background-color: #8B0000">  0.011077</td>
    <td style="background-color: #8B0000">  0.037218</td>
    <td style="background-color: #006400"> -0.026355</td>
    <td style="background-color: #006400"> -0.026123</td>
    <td style="background-color: #006400">  0.000363</td>
    <td style="background-color: #FFD700"> -0.022715</td>
    <td style="background-color: #FFD700">  0.011832</td>
    <td style="background-color: #FFD700"> -0.029748</td>
  </tr>
  <tr>
    <td style="background-color: #8B0000"> -0.046988</td>
    <td style="background-color: #8B0000">  0.018576</td>
    <td style="background-color: #8B0000">  0.013157</td>
    <td style="background-color: #006400">  0.037008</td>
    <td style="background-color: #006400"> -0.000759</td>
    <td style="background-color: #006400"> -0.000307</td>
    <td style="background-color: #FFD700"> -0.029216</td>
    <td style="background-color: #FFD700"> -0.006088</td>
    <td style="background-color: #FFD700"> -0.031105</td>
  </tr>
</table>


(1) the array for the cudnn is a flat array which consists of all kernels and biases.
(2) The general order is sill kernel wieights , recurrent weights and biases, However, the Wi and Wr's order is swapped, meaning the corresponding biases also need to swap.

So, if the weigths stored or extracted in Keras, we need to make some transformation (a bunch of slicing, concatenation, transpose, etc) before useing cuDNN.
Fortunatately, we can find a hiding tool from TF to do the convertion for us, but we still need to do some processing like the slicing, reorder to locate the different weights and biases.
code:


## LSTM
it = σ(Wixt + Riht-1 + bWi + bRi)
ft = σ(Wfxt + Rfht-1 + bWf + bRf)
ot = σ(Woxt + Roht-1 + bWo + bRo)
c't = tanh(Wcxt + Rcht-1 + bWc + bRc)
ct = ft ◦ ct-1 + it ◦ c't
ht = ot ◦ tanh(ct)

Keras Kernel Weights: 
<!---
#90EE90 = lightgreen 
#CD5C5C = indianred
#FFFFE0 = lightyellow
#ADD8E6 = lightblue
#008000 = green
#FF0000 = red
#FFFF00 = yellow
#0000FF = blue
#006400 = darkgreen
#8B0000 = darkred
#FFD700 = gold
#00008B = darkblue --->

<table border=0px style="font-size:13px;">
  <tr>
    <td style="background-color: #90EE90">  0.307402</td>
    <td style="background-color: #90EE90"> -0.468454</td>
    <td style="background-color: #90EE90"> -0.571665</td>
    <td style="background-color: #CD5C5C"> -0.406933</td>
    <td style="background-color: #CD5C5C">  0.390397</td>
    <td style="background-color: #CD5C5C">  0.267421</td>
    <td style="background-color: #FFFFE0"> -0.119232</td>
    <td style="background-color: #FFFFE0">  0.018690</td>
    <td style="background-color: #FFFFE0"> -0.560165</td>
    <td style="background-color: #ADD8E6"> -0.202529</td>
    <td style="background-color: #ADD8E6">  0.328128</td>
    <td style="background-color: #ADD8E6"> -0.453909</td>
  </tr>
  <tr>
    <td style="background-color: #90EE90"> -0.309438</td>
    <td style="background-color: #90EE90">  0.163861</td>
    <td style="background-color: #90EE90">  0.202521</td>
    <td style="background-color: #CD5C5C"> -0.397582</td>
    <td style="background-color: #CD5C5C">  0.334114</td>
    <td style="background-color: #CD5C5C"> -0.077433</td>
    <td style="background-color: #FFFFE0"> -0.450064</td>
    <td style="background-color: #FFFFE0">  0.124535</td>
    <td style="background-color: #FFFFE0">  0.564949</td>
    <td style="background-color: #ADD8E6"> -0.374840</td>
    <td style="background-color: #ADD8E6">  0.154384</td>
    <td style="background-color: #ADD8E6"> -0.276332</td>
  </tr>
</table>


Keras Recurrent Weights: 
<table border=0px style="font-size:13px;">
  <tr>
    <td style="background-color: #008000"> -0.338174</td>
    <td style="background-color: #008000"> -0.019739</td>
    <td style="background-color: #008000">  0.702717</td>
    <td style="background-color: #FF0000">  0.173684</td>
    <td style="background-color: #FF0000"> -0.237763</td>
    <td style="background-color: #FF0000"> -0.398269</td>
    <td style="background-color: #FFFF00"> -0.122475</td>
    <td style="background-color: #FFFF00">  0.061238</td>
    <td style="background-color: #FFFF00">  0.148485</td>
    <td style="background-color: #0000FF">  0.106563</td>
    <td style="background-color: #0000FF">  0.249839</td>
    <td style="background-color: #0000FF">  0.177616</td>
  </tr>
  <tr>
    <td style="background-color: #008000"> -0.202324</td>
    <td style="background-color: #008000"> -0.259554</td>
    <td style="background-color: #008000">  0.264483</td>
    <td style="background-color: #FF0000"> -0.176437</td>
    <td style="background-color: #FF0000">  0.164398</td>
    <td style="background-color: #FF0000">  0.278202</td>
    <td style="background-color: #FFFF00">  0.151397</td>
    <td style="background-color: #FFFF00">  0.039010</td>
    <td style="background-color: #FFFF00">  0.493140</td>
    <td style="background-color: #0000FF"> -0.168453</td>
    <td style="background-color: #0000FF"> -0.028650</td>
    <td style="background-color: #0000FF"> -0.623991</td>
  </tr>
  <tr>
    <td style="background-color: #008000"> -0.114759</td>
    <td style="background-color: #008000"> -0.399628</td>
    <td style="background-color: #008000"> -0.053830</td>
    <td style="background-color: #FF0000">  0.166763</td>
    <td style="background-color: #FF0000">  0.137982</td>
    <td style="background-color: #FF0000"> -0.207373</td>
    <td style="background-color: #FFFF00">  0.150091</td>
    <td style="background-color: #FFFF00">  0.639458</td>
    <td style="background-color: #FFFF00"> -0.216613</td>
    <td style="background-color: #0000FF">  0.321846</td>
    <td style="background-color: #0000FF"> -0.380653</td>
    <td style="background-color: #0000FF"> -0.086838</td>
  </tr>
</table>

Keras Biases: 
<table border=0px style="font-size:13px;">
  <tr>
    <td style="background-color: #006400">  0.049217</td>
    <td style="background-color: #006400">  0.048934</td>
    <td style="background-color: #006400">  0.007049</td>
    <td style="background-color: #8B0000">  1.000000</td>
    <td style="background-color: #8B0000">  1.000000</td>
    <td style="background-color: #8B0000">  1.000000</td>
    <td style="background-color: #FFD700"> -0.020231</td>
    <td style="background-color: #FFD700">  0.046288</td>
    <td style="background-color: #FFD700"> -0.007113</td>
    <td style="background-color: #00008B"> -0.013948</td>
    <td style="background-color: #00008B"> -0.023413</td>
    <td style="background-color: #00008B"> -0.001040</td>
  </tr>
</table>
<!---
#90EE90 = lightgreen 
#CD5C5C = indianred
#FFFFE0 = lightyellow
#ADD8E6 = lightblue
#008000 = green
#FF0000 = red
#FFFF00 = yellow
#0000FF = blue
#006400 = darkgreen
#8B0000 = darkred
#FFD700 = gold
#00008B = darkblue --->

CUDNN Weights:
<table border=0px style="font-size:13px;">
  <tr>
    <td style="background-color: #90EE90">  0.307402</td>
    <td style="background-color: #90EE90"> -0.309438</td>
    <td style="background-color: #90EE90"> -0.468454</td>
    <td style="background-color: #90EE90">  0.163861</td>
    <td style="background-color: #90EE90"> -0.571665</td>
    <td style="background-color: #90EE90">  0.202521</td>
    <td style="background-color: #CD5C5C"> -0.406933</td>
    <td style="background-color: #CD5C5C"> -0.397582</td>
    <td style="background-color: #CD5C5C">  0.390397</td>
    <td style="background-color: #CD5C5C">  0.334114</td>
    <td style="background-color: #CD5C5C">  0.267421</td>
    <td style="background-color: #CD5C5C"> -0.077433</td>
    <td style="background-color: #FFFFE0"> -0.119232</td>
    <td style="background-color: #FFFFE0"> -0.450064</td>
    <td style="background-color: #FFFFE0">  0.018690</td>
    <td style="background-color: #FFFFE0">  0.124535</td>
    <td style="background-color: #FFFFE0"> -0.560165</td>
    <td style="background-color: #FFFFE0">  0.564949</td>
    <td style="background-color: #ADD8E6"> -0.202529</td>
    <td style="background-color: #ADD8E6"> -0.374840</td>
    <td style="background-color: #ADD8E6">  0.328128</td>
    <td style="background-color: #ADD8E6">  0.154384</td>
    <td style="background-color: #ADD8E6"> -0.453909</td>
    <td style="background-color: #ADD8E6"> -0.276332</td>
    <td style="background-color: #008000"> -0.338174</td>
    <td style="background-color: #008000"> -0.202324</td>
    <td style="background-color: #008000"> -0.114759</td>
    <td style="background-color: #008000"> -0.019739</td>
    <td style="background-color: #008000"> -0.259554</td>
    <td style="background-color: #008000"> -0.399628</td>
    <td style="background-color: #008000">  0.702717</td>
    <td style="background-color: #008000">  0.264483</td>
    <td style="background-color: #008000"> -0.053830</td>
    <td style="background-color: #FF0000">  0.173684</td>
    <td style="background-color: #FF0000"> -0.176437</td>
    <td style="background-color: #FF0000">  0.166763</td>
    <td style="background-color: #FF0000"> -0.237763</td>
    <td style="background-color: #FF0000">  0.164398</td>
    <td style="background-color: #FF0000">  0.137982</td>
    <td style="background-color: #FF0000"> -0.398269</td>
    <td style="background-color: #FF0000">  0.278202</td>
    <td style="background-color: #FF0000"> -0.207373</td>
    <td style="background-color: #FFFFE0"> -0.122475</td>
    <td style="background-color: #FFFFE0">  0.151397</td>
    <td style="background-color: #FFFFE0">  0.150091</td>
    <td style="background-color: #FFFFE0">  0.061238</td>
    <td style="background-color: #FFFFE0">  0.039010</td>
    <td style="background-color: #FFFFE0">  0.639458</td>
    <td style="background-color: #FFFFE0">  0.148485</td>
    <td style="background-color: #FFFFE0">  0.493140</td>
    <td style="background-color: #FFFFE0"> -0.216613</td>
    <td style="background-color: #0000FF">  0.106563</td>
    <td style="background-color: #0000FF"> -0.168453</td>
    <td style="background-color: #0000FF">  0.321846</td>
    <td style="background-color: #0000FF">  0.249839</td>
    <td style="background-color: #0000FF"> -0.028650</td>
    <td style="background-color: #0000FF"> -0.380653</td>
    <td style="background-color: #0000FF">  0.177616</td>
    <td style="background-color: #0000FF"> -0.623991</td>
    <td style="background-color: #0000FF"> -0.086838</td>
    <td style="background-color: #006400">  0.000000</td>
    <td style="background-color: #006400">  0.000000</td>
    <td style="background-color: #006400">  0.000000</td>
    <td style="background-color: #8B0000">  0.000000</td>
    <td style="background-color: #8B0000">  0.000000</td>
    <td style="background-color: #8B0000">  0.000000</td>
    <td style="background-color: #FFD700">  0.000000</td>
    <td style="background-color: #FFD700">  0.000000</td>
    <td style="background-color: #FFD700">  0.000000</td>
    <td style="background-color: #00008B">  0.000000</td>
    <td style="background-color: #00008B">  0.000000</td>
    <td style="background-color: #00008B">  0.000000</td>
    <td style="background-color: #006400">  0.049217</td>
    <td style="background-color: #006400">  0.048934</td>
    <td style="background-color: #006400">  0.007049</td>
    <td style="background-color: #8B0000">  1.000000</td>
    <td style="background-color: #8B0000">  1.000000</td>
    <td style="background-color: #8B0000">  1.000000</td>
    <td style="background-color: #FFD700"> -0.020231</td>
    <td style="background-color: #FFD700">  0.046288</td>
    <td style="background-color: #FFD700"> -0.007113</td>
    <td style="background-color: #00008B"> -0.013948</td>
    <td style="background-color: #00008B"> -0.023413</td>
    <td style="background-color: #00008B"> -0.001040</td>
  </tr>
</table>























































































## Reference
* [NVIDIA CUDNN Documentation: cudnnRNNMode_t](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNMode_t)

