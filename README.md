# Imbalance Data Problem


## Requirements
```
sklearn
numpy
```

## SMOTE
SMOTE is a synthetic minority over-sampling technique mentioned in N. V. Chawla, K. W. Bowyer, L. O. Hall and W. P. Kegelmeyer's paper [SMOTE: Synthetic Minority Over-sampling Technique][1]

```
Parameters
----------
sample:     2D (numpy)array
            minority class samples
N:          Integer
            amount of SMOTE N%
k:          Integer
            number of nearest neighbors k
            k <= number of minority class samples
Attributes
----------
newIndex    Integer
            keep a count of number of synthetic samples
            initialize as 0
synthetic   2D array
            array for synthetic samples
neighbors   K-Nearest Neighbors model

```

The corresponding code is in [smote.py][2]. 

### Example
```
from smote import Smote
import numpy as np

X = np.array([[1, 0.7], [0.95, 0.76], [0.98, 0.85], [0.95, 0.78], [1.12, 0.81]])
s = Smote(sample=X, N=300, k=3)
s.over_sampling()
print(s.synthetic)

```
The output will be:
```
[[0.9688157377661356, 0.7470434369118096], [0.970373970826427, 0.7203406632716296], [0.955180350748186, 0.7209519703266685], [0.95, 0.76], [0.9603507618011522, 0.7093355880188698], [0.95, 0.76], [0.98, 0.85], [0.98, 0.85], [0.9767000397651937, 0.8023105914068543], [0.95, 0.78], [0.95, 0.78], [0.9536226582758756, 0.8380845147770741], [1.025027934535906, 0.8276733346832177], [1.0691988855686414, 0.8064896755773396], [1.0457470065562635, 0.7305641034293823]]

```

### Visualization
Suppose the blue triangles are majority class data, the green triangles are minority class data. 
With synthetic minority over-sampling, the red dots are the synthetic samples we generated.

![](https://github.com/zhu-y/Imbalanced-data/blob/master/image/smote_example_1.png)


[1]: https://arxiv.org/pdf/1106.1813.pdf
[2]: https://github.com/zhu-y/Imbalance-Data/blob/master/smote.py
