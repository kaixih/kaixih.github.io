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
Recurrent Neural Network (RNN) is widely used in AI applications of handwriting
recognition, speech recognition, etc. It essentially consists of a series of
matrix-vector multiplications and there are two popular gating mechanisms: GRU
and LSTM. Tensorflow Keras provides high-level APIs for such operations, which
are simple to use and productive, because it can handle the
parameter creation, initialization, and other preprocessing before calling the
actual libraries, where Tensorflow Keras adopts CUDNN as the backend for GPUs. 
In comparison, there are demands that people would like to use CUDNN directly in
their projects for more efficiency and better control of the data flow. This
might need porting the Keras code or simply viewing its outputs as reference. However,
Keras and CUDNN takes different means to deal with the parameters, leading to
the different layouts of parameters. This often causes a bit of confusion
when developers work on both APIs.


This post will check the GRU and LSTM layers from Keras, especially focusing on
how the parameters are organized in Keras and what transformations are needed to
make the parameters compatible for CUDNN. This post assumes people have
sufficient background of RNN. The equations used below are borrowed from the NVIDIA
CUDNN documentation.

## GRU

GRU equations |
--- |
i<sub>t</sub> = σ(W<sub>i</sub>x<sub>t</sub> + R<sub>i</sub>h<sub>t-1</sub> + b<sub>W<sub>i</sub></sub> + b<sub>R<sub>i</sub></sub>) |
r<sub>t</sub> = σ(W<sub>r</sub>x<sub>t</sub> + R<sub>r</sub>h<sub>t-1</sub> + b<sub>W<sub>r</sub></sub> + b<sub>R<sub>r</sub></sub>) |
h'<sub>t</sub> = tanh(W<sub>h</sub>x<sub>t</sub> + r<sub>t</sub>◦(R<sub>h</sub>h<sub>t-1</sub> + b<sub>R<sub>h</sub></sub>) + b<sub>W<sub>h</sub></sub>) |
h<sub>t</sub> = (1 - i<sub>t</sub>) ◦ h'<sub>t</sub> + i<sub>t</sub> ◦ h<sub>t-1</sub> |

In these equations, the parameters consisit of three kernel weights/biases
(W<sub>i</sub>, W<sub>r</sub>, W<sub>h</sub>, b<sub>W<sub>i</sub></sub>,
b<sub>W<sub>r</sub></sub>, b<sub>W<sub>h</sub></sub>) and three recurrent
weights/biases (R<sub>i</sub>, R<sub>r</sub>, R<sub>h</sub>,
b<sub>R<sub>i</sub></sub>, b<sub>R<sub>r</sub></sub>,
b<sub>R<sub>h</sub></sub>). These two sets of weights/biases pairs indicate that
we apply bias addition whether the weight-input multiplication is for the layer
input x<sub>t</sub> or for the recurrent input h<sub>t-1</sub>. In fact, the
above equations represent a "double bias" mode categorized by CUDNN and
apparently there are two other bias modes that apply bias addition only to layer
input or only to recurrent input.

Here I would like to point out the x<sub>t</sub> and h<sub>t-1</sub> are both column
vectors in the shape of (inputSize, 1) and (hiddenSize, 1) respectively to make
the matrix-vector multiplication possible. Therefore, the shape of each W
weights is (hiddenSize, inputSize) and the shape of each R weights is (hiddenSize,
hiddenSize), while the biases are always in (hiddenSize, 1).

The above discussion is based on CUDNN formula; by constrast, Keras takes a
slightly different way to interpret the weight-input multiplication, which is
more like x<sub>t</sub><sup>T</sup>W<sup>T</sup> or
h<sub>t-1</sub><sup>T</sup>R<sup>T</sup>. That way, the weights in Keras are
stored in a transposed style compared to CUDNN: the W weights are (inputSize,
hiddenSize) and the R weights are (hiddenSize, hiddenSize). As for the specific
implementation under the hood, CUDNN stores all weights and biases all together
as a single flatten array, while Keras uses three arrays:
1. Kernel weights: a (inputSize, 3 x hiddenSize) matrix containing the
   concatenated three W weights.
2. Recurrent weights: a (hiddenSize, 3 x hiddenSize) matrix containing the
   concatenated three R weights.
3. Biases: a (2, 3 x hiddenSize) matrix where one row is the concatenated W biases
   and another is for R biases.

The following python code is to output the parameters stored in Keras GRU layer:
```python
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.python.keras.layers import recurrent_v2

batch_size = 1
seq_len = 2
input_size = 2
hidden_size = 3

tf.random.set_seed(seed=1)
x = tf.random.uniform((seq_len, batch_size, input_size))
gru = layers.GRU(hidden_size, time_major=True,
                 return_sequences=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='random_uniform')
y = gru(x)

np.set_printoptions(
    formatter={'float': lambda x: "{0:0.6f}".format(x)})

print("Keras Kernel Weights:", gru.get_weights()[0])
print("Keras Recurrent Weights:", gru.get_weights()[1])
print("Keras Biases:", gru.get_weights()[2])

```
Here I don't copy/paste the outputs but visualize them with three colors to
represent different types of weights/biases:
* Green for _r_ weights/biases
* Red for _i_ weights/biases
* Yellow for _h_ weights/biases

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
<table border=0px style="font-size:12px;">
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
<table border=0px style="font-size:12px;">
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
<table border=0px style="font-size:12px;">
  <tr>
    <td style="background-color: #006400;color: white"> -0.026355</td>
    <td style="background-color: #006400;color: white"> -0.026123</td>
    <td style="background-color: #006400;color: white">  0.000363</td>
    <td style="background-color: #8B0000;color: white">  0.027354</td>
    <td style="background-color: #8B0000;color: white">  0.011077</td>
    <td style="background-color: #8B0000;color: white">  0.037218</td>
    <td style="background-color: #FFD700"> -0.022715</td>
    <td style="background-color: #FFD700">  0.011832</td>
    <td style="background-color: #FFD700"> -0.029748</td>
  </tr>
  <tr>
    <td style="background-color: #006400;color: white">  0.037008</td>
    <td style="background-color: #006400;color: white"> -0.000759</td>
    <td style="background-color: #006400;color: white"> -0.000307</td>
    <td style="background-color: #8B0000;color: white"> -0.046988</td>
    <td style="background-color: #8B0000;color: white">  0.018576</td>
    <td style="background-color: #8B0000;color: white">  0.013157</td>
    <td style="background-color: #FFD700"> -0.029216</td>
    <td style="background-color: #FFD700"> -0.006088</td>
    <td style="background-color: #FFD700"> -0.031105</td>
  </tr>
</table>

Moreover, I add some printfs in the backend to trace the inputs/outputs of CUDNN
calls. The parameters directly taken in by CUDNN are as below and I also tint them
with the three colors.

CUDNN Parameters (Weights and Biases):
<table border=0px style="font-size:12px;">
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
    <td style="background-color: #FFFF00">  0.278958</td>
    <td style="background-color: #FFFF00">  0.519950</td>
    <td style="background-color: #FFFF00"> -0.030054</td>
    <td style="background-color: #FFFF00">  0.004496</td>
    <td style="background-color: #FFFF00"> -0.085155</td>
    <td style="background-color: #FFFF00"> -0.501773</td>
    <td style="background-color: #FFFF00">  0.346469</td>
    <td style="background-color: #FFFF00"> -0.509539</td>
    <td style="background-color: #FFFF00"> -0.211831</td>
  </tr>
  <tr>
    <td style="background-color: #8B0000;color: white">  0.027354</td>
    <td style="background-color: #8B0000;color: white">  0.011077</td>
    <td style="background-color: #8B0000;color: white">  0.037218</td>
    <td style="background-color: #006400;color: white"> -0.026355</td>
    <td style="background-color: #006400;color: white"> -0.026123</td>
    <td style="background-color: #006400;color: white">  0.000363</td>
    <td style="background-color: #FFD700"> -0.022715</td>
    <td style="background-color: #FFD700">  0.011832</td>
    <td style="background-color: #FFD700"> -0.029748</td>
  </tr>
  <tr>
    <td style="background-color: #8B0000;color: white"> -0.046988</td>
    <td style="background-color: #8B0000;color: white">  0.018576</td>
    <td style="background-color: #8B0000;color: white">  0.013157</td>
    <td style="background-color: #006400;color: white">  0.037008</td>
    <td style="background-color: #006400;color: white"> -0.000759</td>
    <td style="background-color: #006400;color: white"> -0.000307</td>
    <td style="background-color: #FFD700"> -0.029216</td>
    <td style="background-color: #FFD700"> -0.006088</td>
    <td style="background-color: #FFD700"> -0.031105</td>
  </tr>
</table>


By comparing the parameters in Keras and CUDNN, we can observe three
differences:
1. CUDNN uses a single flat array, while Keras uses three separate arrays.
2. The weights are transposed between CUDNN and Keras.
3. The _i_ and _r_ weights/biases are swapped between CUDNN and Keras.

So, if one is given some weights/biases extracted from Keras and their task is
to use them in CUDNN, they have to perform a series of preprocessing on them,
such as slicing, permuting, tranposing, concatenation, etc. Fortunately, we can
find a "secret" tool from Tensorflow Keras that can partially help over the
preprocessing:
```python
params = recurrent_v2._canonical_to_params(
    weights=[
        gru.get_weights()[0][:, hidden_size:hidden_size * 2],
        gru.get_weights()[0][:, :hidden_size],
        gru.get_weights()[0][:, hidden_size * 2:],
        gru.get_weights()[1][:, hidden_size:hidden_size * 2],
        gru.get_weights()[1][:, :hidden_size],
        gru.get_weights()[1][:, hidden_size * 2:],
    ],
    biases=[
        gru.get_weights()[2][0][hidden_size:hidden_size * 2],
        gru.get_weights()[2][0][:hidden_size],
        gru.get_weights()[2][0][hidden_size * 2:hidden_size * 3],
        gru.get_weights()[2][1][hidden_size:hidden_size * 2],
        gru.get_weights()[2][1][:hidden_size],
        gru.get_weights()[2][1][hidden_size * 2:hidden_size * 3],
    ],
    shape=tf.constant([-1]),
    transpose_weights=True)
print("CUDNN-equivalent Params:", params)
```
Note, as we can see, though it is a single function call, we still need to do
most of the slicing operations and don't forget to swap the _i_ and _r_
(i.e., the 0th and 1st) weights/biases. 


## LSTM

LSTM equations |
--- |
i<sub>t</sub> = σ(W<sub>i</sub>x<sub>t</sub> + R<sub>i</sub>h<sub>t-1</sub> + b<sub>W<sub>i</sub></sub> + b<sub>R<sub>i</sub></sub>) |
f<sub>t</sub> = σ(W<sub>f</sub>x<sub>t</sub> + R<sub>f</sub>h<sub>t-1</sub> + b<sub>W<sub>f</sub></sub> + b<sub>R<sub>f</sub></sub>) |
o<sub>t</sub> = σ(W<sub>o</sub>x<sub>t</sub> + R<sub>o</sub>h<sub>t-1</sub> + b<sub>W<sub>o</sub></sub> + b<sub>R<sub>o</sub></sub>) |
c'<sub>t</sub> = tanh(W<sub>c</sub>x<sub>t</sub> + R<sub>c</sub>h<sub>t-1</sub> + b<sub>W<sub>c</sub></sub> + b<sub>R<sub>c</sub></sub>) |
c<sub>t</sub> = f<sub>t</sub> ◦ c<sub>t-1</sub> + i<sub>t</sub> ◦ c'<sub>t</sub> |
h<sub>t</sub> = o<sub>t</sub> ◦ tanh(c<sub>t</sub>) |

It would be a lot easier to understand the LSTM params format when we have learned it in the GRU.
The LSTM equations show that there are four kernel weights/biases
(W<sub>i</sub>, W<sub>f</sub>, W<sub>o</sub>, W<sub>c</sub>,
b<sub>W<sub>i</sub></sub>, b<sub>W<sub>f</sub></sub>, b<sub>W<sub>o</sub></sub>,
b<sub>W<sub>c</sub></sub>) and four recurrent weights/biases (R<sub>i</sub>,
R<sub>f</sub>, R<sub>o</sub>, R<sub>c</sub>, b<sub>R<sub>i</sub></sub>,
b<sub>R<sub>f</sub></sub>, b<sub>R<sub>o</sub></sub>,
b<sub>R<sub>c</sub></sub>). The shapes of the weights and biases are
same with those in GRU case.

Still, the two sets of weights/biases in the equations demonstrate the "double
bias" mode used by CUDNN. In contrast, Keras adopts a mode more close to the
"single bias" mode that only performs the bias addition once for each equation
above. So, considering the transposed weights and "single bias" mode, the three Keras
LSTM arrays are:
1. Kernel weights: a (inputSize, 4 x hiddenSize) matrix containing the
   concatenated four W weights.
2. Recurrent weights: a (hiddenSize, 4 x hiddenSize) matrix containing the
   concatenated four R weights.
3. Biases: a (4 x hiddenSize) vector containing the concatenated biases.

The following python code is to output the parameters stored in Keras LSTM layer:
```python
lstm = layers.LSTM(hidden_size, time_major=True,
                   return_sequences=True,
                   kernel_initializer='glorot_uniform',
                   recurrent_initializer='orthogonal',
                   bias_initializer='random_uniform')
y = lstm(x)
print("Keras Kernel Weights:", lstm.get_weights()[0])
print("Keras Recurrent Weights:", lstm.get_weights()[1])
print("Keras Biases:", lstm.get_weights()[2])
```
Here I visualize the outputs with four colors to
represent different types of weights/biases:
* Green for _i_ weights/biases
* Red for _f_ weights/biases
* Yellow for _o_ weights/biases
* Blue for _c_ weights/biases

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

<table border=0px style="font-size:12px;">
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
  </tr>
</table>
<table border=0px style="font-size:12px;">
  <tr>
    <td style="background-color: #ADD8E6"> -0.202529</td>
    <td style="background-color: #ADD8E6">  0.328128</td>
    <td style="background-color: #ADD8E6"> -0.453909</td>
  </tr>
  <tr>
    <td style="background-color: #ADD8E6"> -0.374840</td>
    <td style="background-color: #ADD8E6">  0.154384</td>
    <td style="background-color: #ADD8E6"> -0.276332</td>
  </tr>
</table>


Keras Recurrent Weights: 
<table border=0px style="font-size:12px;">
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
  </tr>
</table>
<table border=0px style="font-size:12px;">
  <tr>
    <td style="background-color: #0000FF;color: white">  0.106563</td>
    <td style="background-color: #0000FF;color: white">  0.249839</td>
    <td style="background-color: #0000FF;color: white">  0.177616</td>
  </tr>
  <tr>
    <td style="background-color: #0000FF;color: white"> -0.168453</td>
    <td style="background-color: #0000FF;color: white"> -0.028650</td>
    <td style="background-color: #0000FF;color: white"> -0.623991</td>
  </tr>
  <tr>
    <td style="background-color: #0000FF;color: white">  0.321846</td>
    <td style="background-color: #0000FF;color: white"> -0.380653</td>
    <td style="background-color: #0000FF;color: white"> -0.086838</td>
  </tr>
</table>

Keras Biases: 
<table border=0px style="font-size:12px;">
  <tr>
    <td style="background-color: #006400;color: white">  0.049217</td>
    <td style="background-color: #006400;color: white">  0.048934</td>
    <td style="background-color: #006400;color: white">  0.007049</td>
    <td style="background-color: #8B0000;color: white">  1.000000</td>
    <td style="background-color: #8B0000;color: white">  1.000000</td>
    <td style="background-color: #8B0000;color: white">  1.000000</td>
    <td style="background-color: #FFD700"> -0.020231</td>
    <td style="background-color: #FFD700">  0.046288</td>
    <td style="background-color: #FFD700"> -0.007113</td>
  </tr>
</table>
<table border=0px style="font-size:12px;">
  <tr>
    <td style="background-color: #00008B;color: white"> -0.013948</td>
    <td style="background-color: #00008B;color: white"> -0.023413</td>
    <td style="background-color: #00008B;color: white"> -0.001040</td>
  </tr>
</table>

Similarly, I tint the parameters taken in by CUDNN with the four colors.

CUDNN Parameters (Weights and Biases):
<table border=0px style="font-size:12px;">
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
  </tr>
  <tr>
    <td style="background-color: #CD5C5C">  0.334114</td>
    <td style="background-color: #CD5C5C">  0.267421</td>
    <td style="background-color: #CD5C5C"> -0.077433</td>
    <td style="background-color: #FFFFE0"> -0.119232</td>
    <td style="background-color: #FFFFE0"> -0.450064</td>
    <td style="background-color: #FFFFE0">  0.018690</td>
    <td style="background-color: #FFFFE0">  0.124535</td>
    <td style="background-color: #FFFFE0"> -0.560165</td>
    <td style="background-color: #FFFFE0">  0.564949</td>
  </tr>
  <tr>
    <td style="background-color: #ADD8E6"> -0.202529</td>
    <td style="background-color: #ADD8E6"> -0.374840</td>
    <td style="background-color: #ADD8E6">  0.328128</td>
    <td style="background-color: #ADD8E6">  0.154384</td>
    <td style="background-color: #ADD8E6"> -0.453909</td>
    <td style="background-color: #ADD8E6"> -0.276332</td>
    <td style="background-color: #008000"> -0.338174</td>
    <td style="background-color: #008000"> -0.202324</td>
    <td style="background-color: #008000"> -0.114759</td>
  </tr>
  <tr>
    <td style="background-color: #008000"> -0.019739</td>
    <td style="background-color: #008000"> -0.259554</td>
    <td style="background-color: #008000"> -0.399628</td>
    <td style="background-color: #008000">  0.702717</td>
    <td style="background-color: #008000">  0.264483</td>
    <td style="background-color: #008000"> -0.053830</td>
    <td style="background-color: #FF0000">  0.173684</td>
    <td style="background-color: #FF0000"> -0.176437</td>
    <td style="background-color: #FF0000">  0.166763</td>
  </tr>
  <tr>
    <td style="background-color: #FF0000"> -0.237763</td>
    <td style="background-color: #FF0000">  0.164398</td>
    <td style="background-color: #FF0000">  0.137982</td>
    <td style="background-color: #FF0000"> -0.398269</td>
    <td style="background-color: #FF0000">  0.278202</td>
    <td style="background-color: #FF0000"> -0.207373</td>
    <td style="background-color: #FFFF00"> -0.122475</td>
    <td style="background-color: #FFFF00">  0.151397</td>
    <td style="background-color: #FFFF00">  0.150091</td>
  </tr>
  <tr>
    <td style="background-color: #FFFF00">  0.061238</td>
    <td style="background-color: #FFFF00">  0.039010</td>
    <td style="background-color: #FFFF00">  0.639458</td>
    <td style="background-color: #FFFF00">  0.148485</td>
    <td style="background-color: #FFFF00">  0.493140</td>
    <td style="background-color: #FFFF00"> -0.216613</td>
    <td style="background-color: #0000FF;color: white">  0.106563</td>
    <td style="background-color: #0000FF;color: white"> -0.168453</td>
    <td style="background-color: #0000FF;color: white">  0.321846</td>
  </tr>
  <tr>
    <td style="background-color: #0000FF;color: white">  0.249839</td>
    <td style="background-color: #0000FF;color: white"> -0.028650</td>
    <td style="background-color: #0000FF;color: white"> -0.380653</td>
    <td style="background-color: #0000FF;color: white">  0.177616</td>
    <td style="background-color: #0000FF;color: white"> -0.623991</td>
    <td style="background-color: #0000FF;color: white"> -0.086838</td>
    <td style="background-color: #006400;color: white">  0.000000</td>
    <td style="background-color: #006400;color: white">  0.000000</td>
    <td style="background-color: #006400;color: white">  0.000000</td>
  </tr>
  <tr>
    <td style="background-color: #8B0000;color: white">  0.000000</td>
    <td style="background-color: #8B0000;color: white">  0.000000</td>
    <td style="background-color: #8B0000;color: white">  0.000000</td>
    <td style="background-color: #FFD700">  0.000000</td>
    <td style="background-color: #FFD700">  0.000000</td>
    <td style="background-color: #FFD700">  0.000000</td>
    <td style="background-color: #00008B;color: white">  0.000000</td>
    <td style="background-color: #00008B;color: white">  0.000000</td>
    <td style="background-color: #00008B;color: white">  0.000000</td>
  </tr>
  <tr>
    <td style="background-color: #006400;color: white">  0.049217</td>
    <td style="background-color: #006400;color: white">  0.048934</td>
    <td style="background-color: #006400;color: white">  0.007049</td>
    <td style="background-color: #8B0000;color: white">  1.000000</td>
    <td style="background-color: #8B0000;color: white">  1.000000</td>
    <td style="background-color: #8B0000;color: white">  1.000000</td>
    <td style="background-color: #FFD700"> -0.020231</td>
    <td style="background-color: #FFD700">  0.046288</td>
    <td style="background-color: #FFD700"> -0.007113</td>
  </tr>
  <tr>
    <td style="background-color: #00008B;color: white"> -0.013948</td>
    <td style="background-color: #00008B;color: white"> -0.023413</td>
    <td style="background-color: #00008B;color: white"> -0.001040</td>
  </tr>
</table>

By comparing the parameters in Keras and CUDNN, we can still observe three major
differences:
1. CUDNN uses a single flat array, while Keras uses four separate arrays.
2. The weights are transposed between CUDNN and Keras.
3. The order of weights/biases are same between CUDNN and Keras, but the bias part
   is padded with (4 x hiddenSize) zeros to fill the gaps of the "double bias"
   (CUDNN) and "single bias" (Keras) modes.

Now, we take advantage of the "secret" function to convert the Keras parameters
to be CUDNN compatible:
```python
params = recurrent_v2._canonical_to_params(
    weights=[
        lstm.get_weights()[0][:, :hidden_size],
        lstm.get_weights()[0][:, hidden_size:hidden_size * 2],
        lstm.get_weights()[0][:, hidden_size * 2:hidden_size * 3],
        lstm.get_weights()[0][:, hidden_size * 3:],
        lstm.get_weights()[1][:, :hidden_size],
        lstm.get_weights()[1][:, hidden_size:hidden_size * 2],
        lstm.get_weights()[1][:, hidden_size * 2:hidden_size * 3],
        lstm.get_weights()[1][:, hidden_size * 3:],
    ],
    biases=[
        tf.zeros((hidden_size,)),
        tf.zeros((hidden_size,)),
        tf.zeros((hidden_size,)),
        tf.zeros((hidden_size,)),
        lstm.get_weights()[2][:hidden_size],
        lstm.get_weights()[2][hidden_size:hidden_size * 2],
        lstm.get_weights()[2][hidden_size * 2:hidden_size * 3],
        lstm.get_weights()[2][hidden_size * 3:hidden_size * 4],
    ],
    shape=tf.constant([-1]),
    transpose_weights=True)
print("CUDNN-equivalent Params:", params)
```
Note, for LSTM, we don't need to change orders of the four weights/biases but
do need to pad one row of biases to be zeros.

## Reference
* [NVIDIA CUDNN Documentation: cudnnRNNMode_t](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNMode_t)
* [Tensorflow Keras Layers GRU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)
* [Tensorflow Keras Layers LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)

