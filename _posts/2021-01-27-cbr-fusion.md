---
layout: posts
title:  "Demystifying the Conv-Bias-ReLU Fusion"
published: false
author: kaixi_hou
search                   : true
search_full_content      : true
search_provider          : google
comments: true
---
(Under construction)
## Introduction
## Convolution Pattern
### Convolution Forward
Convolution equations |
--- |
y<sub>11</sub> = w<sub>11</sub>x<sub>11</sub> + w<sub>12</sub>x<sub>12</sub> + w<sub>21</sub>x<sub>21</sub> + w<sub>22</sub>x<sub>22</sub> |
y<sub>12</sub> = w<sub>11</sub>x<sub>12</sub> + w<sub>12</sub>x<sub>13</sub> + w<sub>21</sub>x<sub>22</sub> + w<sub>22</sub>x<sub>23</sub> |
y<sub>21</sub> = w<sub>11</sub>x<sub>21</sub> + w<sub>12</sub>x<sub>22</sub> + w<sub>21</sub>x<sub>31</sub> + w<sub>22</sub>x<sub>32</sub> |
y<sub>22</sub> = w<sub>11</sub>x<sub>22</sub> + w<sub>12</sub>x<sub>23</sub> + w<sub>21</sub>x<sub>32</sub> + w<sub>22</sub>x<sub>33</sub> |



### Convolution Backward
Suppose e is the error (or cost/loss) and dy is same with ∂e/∂y.

Roughly speaking, dw = ∂e/∂w = (∂e/∂y)(∂y/∂w) = dy⋅x
Weight gradient equations |
--- |
dw<sub>11</sub> = dy<sub>11</sub>x<sub>11</sub> + dy<sub>12</sub>x<sub>12</sub> + dy<sub>21</sub>x<sub>21</sub> + dy<sub>22</sub>x<sub>22</sub> |
dw<sub>12</sub> = dy<sub>11</sub>x<sub>12</sub> + dy<sub>12</sub>x<sub>13</sub> + dy<sub>21</sub>x<sub>22</sub> + dy<sub>22</sub>x<sub>23</sub> |
dw<sub>21</sub> = dy<sub>11</sub>x<sub>21</sub> + dy<sub>12</sub>x<sub>22</sub> + dy<sub>21</sub>x<sub>31</sub> + dy<sub>22</sub>x<sub>32</sub> |
dw<sub>22</sub> = dy<sub>11</sub>x<sub>22</sub> + dy<sub>12</sub>x<sub>23</sub> + dy<sub>21</sub>x<sub>32</sub> + dy<sub>22</sub>x<sub>33</sub> |

Roughly speaking, dx = ∂e/∂x = (∂e/∂y)(∂y/∂x) = dy⋅w
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
### Convolution in a Graph

## BiasAdd Pattern
### BiasAdd Forward
BiasAdd equations |
--- |
y = x + b |
### BiasAdd Backward
db = ∂e/∂b = (∂e/∂y)(∂y/∂b) = dy
dx = ∂e/∂x = (∂e/∂y)(∂y/∂x) = dy

### BiasAdd in a Graph

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

## Putting Them All Together
