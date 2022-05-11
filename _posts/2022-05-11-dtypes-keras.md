---
layout: posts
title:  "Expected Data Types in Mixed Precision Cheatsheet"
published: true
author: kaixi_hou
search                   : true
search_full_content      : true
search_provider          : google
comments: true
---

When training neural networks with the Keras API, we care about the data types
and computation types since they are relevant to the convergence (numeric
stability) and performance (memory footprint and computation efficiency). There
are multiple "knobs" that we can turn to change the types by:
1. Setting `dtype` of input tensors, or explicitly `tf.cast` the tensors;
1. Setting `dtype` of the Keras Layer which defines data type of the layer's
   computations and weights;
1. Using the environment variable `TF_FP16_CONV_USE_FP32_COMPUTE` for the
   computation data type;
1. Using the mixed precision policy globally:
   `mixed_precision.set_global_policy('mixed_float16')`.

This may seem a bit confusing and it may not be clear how different settings
affect each other and what we should expect about the actual weight/output data
type and computation data type.  

Therefore, I am trying to sweep through all possible combinations of the
settings (fp16 or fp32) and the table below summarizes the obtained weight/output
data type and computation data type for them and I hope the examples help.

| Layer | Input    | TF_FP16_CONV_USE _FP32_COMPUTE | Weight | Computation | Output |
|-------------|-------------|-------------------------------|--------------|------------|-----------|
| fp32        | fp32        | 1                             | fp32         | fp32       | fp32      |
| fp32        | fp32        | 0                             | fp32         | fp32       | fp32      |
| fp32        | fp16(⇒fp32) | 1                             | fp32         | fp32       | fp32      |
| fp32        | fp16(⇒fp32) | 0                             | fp32         | fp32       | fp32      |
| fp16        | fp32(⇒fp16) | 1                             | fp16         | fp32       | fp16      |
| fp16        | fp32(⇒fp16) | 0                             | fp16         | fp16       | fp16      |
| fp16        | fp16        | 1                             | fp16         | fp32       | fp16      |
| fp16        | fp16        | 0                             | fp16         | fp16       | fp16      |
| mixed       | fp32(⇒fp16) | 1                             | fp32(⇒fp16)  | fp32       | fp16      |
| mixed       | fp32(⇒fp16) | 0                             | fp32(⇒fp16)  | fp16       | fp16      |
| mixed       | fp16        | 1                             | fp32(⇒fp16)  | fp32       | fp16      |
| mixed       | fp16        | 0                             | fp32(⇒fp16)  | fp16       | fp16      |

A(=>B) means the data is stored in A but will be automatically cast to B.

Basically we can summarize the behavior with four rules:
* The layer dtype determines the actual input/weight dtype.
* "mixed" mainly refers to the weight that can be stored in fp32 but computed in fp16.
* The env var determines the computation dtype when layer dtype is fp16/mixed.
* The output dtype is fp32 only when layer dtype is fp32.
