# NumPy学习笔记

## 入门

### 布尔屏蔽

根据指定条件检索数组元素：

```python
# Boolean masking
import matplotlib.pyplot as plt

a = np.linspace(0, 2 * np.pi, 50)
b = np.sin(a)
plt.plot(a,b)
mask = b >= 0
plt.plot(a[mask], b[mask], 'bo')
mask = (b >= 0) & (a <= np.pi / 2)
plt.plot(a[mask], b[mask], 'go')
plt.show()
```

![image-20211201164315873](C:\Users\surafce book2\AppData\Roaming\Typora\typora-user-images\image-20211201164315873.png)

## NumPy教程

### 数组定义

np.full, np.random.random

np.where np.repeat

np.vstack, np.hstack

np.concatenate

np.r_ np.c_

np.tile

np.intersect1d

np.setdiff1d



## SciPy

### 读入matlab文件

`scipy.io.loadmat`

详见：[Input and output (scipy.io) — SciPy v1.7.1 Manual](https://docs.scipy.org/doc/scipy/reference/io.html)

### 点到直线的距离

`scipy.spatial.distance.pdist`

例程：

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Create the following array where each row is a point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
x = np.array([[0, 1], [1, 0], [2, 0]])
print(x)

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
# and d is the following array:
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
d = squareform(pdist(x, 'euclidean'))
print(d)
```

详见：[scipy.spatial.distance.pdist — SciPy v1.7.1 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html)

## Matplotlib

###  plot

plot可以画图，如

```python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)
plt.show()  # You must call plt.show() to make graphics appear.
```

也可以一次画多条，参考：[pyplot — Matplotlib 2.0.2 documentation](https://matplotlib.org/2.0.2/api/pyplot_api.html)

#### 子图

可以使用`subplot`在同一张图绘制不同的东西。