---
layout: posts
title:  "Moving Mean and Moving Variance In Batch Normalization"
published: false
author: kaixi_hou
search                   : true
search_full_content      : true
search_provider          : google
comments: true
---

## Introduction
## Typical Batch Norm
<p align=center> Figure 1. Typical batch norm in Tensorflow Keras</p>
![Typical Batch Norm](/assets/posts_images/bn_orig.png)

## Fused Batch Norm
<p align=center> Figure 2. Fused batch norm on GPUs</p>
![Fused Batch Norm](/assets/posts_images/bn_fuse.png)

## Batch Norm Backpropagation
<p align=center> Figure 3. Fused batch norm and backpropagation</p>
![Batch Norm Backpropagation](/assets/posts_images/bn_grad.png)

## Synchronized Batch Norm
<p align=center> Figure 4. Synchronized batch norm</p>
![Synchronized Batch Norm](/assets/posts_images/bn_sync.png)

