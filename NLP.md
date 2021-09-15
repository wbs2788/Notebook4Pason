# NLP

## 第一章 绪论

### 课前说明

成绩：平时分 10% 三次小作业 3*15% 大作业 45%

作业：

一：命名实体识别：利用隐马尔科夫模型实现所给数据的命名实体识别 

二：中文分词：最大匹配算法

三：新闻文本分类：机器学习、深度学习

大作业：中文摘要生成

## 第二章 数学基础

### 概率论基础

##### 最大似然估计

来求一个样本集的相关概率密度函数的参数，先假定一个概率分布，再求出假定参数最可能的值。

**例：**设总体$X~N(μ$，$σ^2)$，$μ$，$σ$为未知参数，$X_1,X_2...,X_n$是来自总体$X$的样本，$X_1,X_2...,X_n$是对应的样本值，求$μ$与$σ^2$的最大似然估计值。

**解：**X的概率密度为

![img](https://bkimg.cdn.bcebos.com/formula/3e25c1036e3ebf3cf57b2ab2f6e96b02.svg)

可得似然函数如下：

![img](https://bkimg.cdn.bcebos.com/formula/b553e7994bb9f0ed3445b9df3cd5db8a.svg)

取对数，得

![img](https://bkimg.cdn.bcebos.com/formula/fa506e367fd0edb81eff59121e1e1d71.svg)

令

![img](https://bkimg.cdn.bcebos.com/formula/cf6213fd4dc3505d910e9addec46cf29.svg)

![img](https://bkimg.cdn.bcebos.com/formula/59eb9c3de3306e27452370356846db54.svg)

解得

![img](https://bkimg.cdn.bcebos.com/formula/f1460c9667ee703bfa9887008956613e.svg)

![img](https://bkimg.cdn.bcebos.com/formula/c61ade42d9692367efb621423ee9b05c.svg)

故μ和σ的最大似然估计量分别为

![img](https://bkimg.cdn.bcebos.com/formula/7f869735471aef9ffede1dc35e37a4a3.svg)

![img](https://bkimg.cdn.bcebos.com/formula/74172d8507137882dfb8c1f2ce24a00c.svg)

##### 条件概率、全概率公式

##### 贝叶斯决策理论

思想：已知类条件概率密度参数表达式和先验概率，利用贝叶斯公式转换成后验概率进行决策分类。
$$
P(\omega_i|x) = \frac{p(x|\omega_i)p(\omega_i)}{p(x)}
$$
设w1表示学渣组（类别1），w2表示学霸组（类别2），x=0表示卷面成绩不超过90事件，x=1表示卷面成绩90+事件，U表示试卷总份数。

再设P(wi) 表示两组（类）的份数占比，那么P(w1)=0.5, P(w2)=0.5，即各占一半，此概率被称作先验概率。

再假设通过以往所有的考试信息，得出w1组得分90+的概率为0.2，w2组得分90+的概率为0.8，即P(x=1|w1)=0.2, P(x=1|w2)=0.8，此概率常被称作类的条件概率。它反映两者最本质的区别——这里代表考90+的概率，是分类时最重要的依据。

用P(x=1) 表示w1、w2两组得分90+的总概率，是一个全概率。

最终求的是90+的卷子来自w1、w2两组（类别）的概率，即P(w1|x=1)、P(w2|x=1)，它也是一个条件概率，常被称作后验概率。

### 信息论

##### 熵

$$
H(X) = -\sum_{x\in X}p(x)\log_2 p(x)
$$

##### 联合熵

$$
H(X,Y) = - \sum_{x \in X}\sum_{y \in Y}p(x,y)\log_2p(x,y)
$$

##### 条件熵

$$
H(X|Y) - \sum_{x \in X}\sum_{y \in Y}p(x,y)\log_2p(x|y)
$$

##### 连锁规则

$$
H(X) = H(X,Y) + H(X|Y)
$$

