---
layout: post
title:  "Bias Variance Trade Off!"
date:   2019-10-04 8:40:15 
---
There is a concept in machine learning called Bias Variance trade off that I am going to explain below.

```python
import numpy as np
import pandas as pd
```
<h2>Bias-Variance Decomposition</h2>

<p>
Recall that the squared error can be decomposed into <em>bias</em>, <em>variance</em> and <em>noise</em>: 
$$
    \underbrace{\mathbb{E}[(h_D(\mathbf{x}) - y)^2]}_\mathrm{Error} = \underbrace{\mathbb{E}[(h_D(\mathbf{x})-\bar{h}(\mathbf{x}))^2]}_\mathrm{Variance} + \underbrace{\mathbb{E}[(\bar{h}(\mathbf{x})-\bar{y}(\mathbf{x}))^2]}_\mathrm{Bias} + \underbrace{\mathbb{E}[(\bar{y}(\mathbf{x})-y(\mathbf{x}))^2]}_\mathrm{Noise}\nonumber
$$
    
We will now create a data set for which we can approximately compute this decomposition. 
The function <strong>`toydata`</strong> generates a binary data set with class $1$ and $2$. Both are sampled from Gaussian distributions:
$$
p(\mathbf{x}|y=1)\sim {\mathcal{N}}(0,{I}) \textrm { and } p(\mathbf{x}|y=2)\sim {\mathcal{N}}(\mu_2,{I}),
$$

where $\mathbf{\mu_2}=[1.75, 1.75]^\top$ (the global variable <code>OFFSET</code> $\!=\!1.75$ regulates these values: $\mathbf{\mu_2}=[$<code>OFFSET</code> $, $ <code>OFFSET</code>$]^\top$).
</p>

```python
xTe = np.array([
    [49.308783, 49.620651], 
    [1.705462, 1.885418], 
    [ 51.192402, 50.256330],
    [0.205998, -0.089885],
    [50.853083, 51.833237]])  
yTe = np.array([2, 1, 2, 1, 2])
print (xTe, yTe)
```

    [[49.308783 49.620651]
     [ 1.705462  1.885418]
     [51.192402 50.25633 ]
     [ 0.205998 -0.089885]
     [50.853083 51.833237]] [2 1 2 1 2]
    


```python

```
