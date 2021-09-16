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

#### 最大似然估计

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

#### 条件概率、全概率公式

#### 贝叶斯决策理论

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

#### 熵

$$
H(X) = -\sum_{x\in X}p(x)\log p(x)
$$

#### 联合熵

$$
H(X,Y) = - \sum_{x \in X}\sum_{y \in Y}p(x,y)\log p(x,y)
$$

#### 条件熵

$$
H(X|Y) - \sum_{x \in X}\sum_{y \in Y}p(x,y)\log p(x|y)
$$

#### 连锁规则

$$
H(X,Y)= H(X) + H(Y|X)
$$

#### 熵率

$$
H_{rate} = \frac{1}{n}H(X_{1\dots n})=-\frac{1}{n}\sum p(x_{1\dots n})\log p(x_{1\dots n})
$$

#### 相对熵(KL距离)

$$
D(p||q) = \sum_{x\in X}p(x)\log\frac{p(x)}{q(x)}
$$

用来衡量两个随机分布的距离。

#### 交叉熵

$$
H(X,q) = H(X) + D(p||q)
$$

其中$X\sim p(x)$。

可定义语言$L=(X)$与其模型$q$的交叉熵：
$$
H(L,q) = - \lim_{n\to\infin}\frac{1}{n}\sum p(x_1^n)\log q(x_1^n)
$$
如果语言$L$是稳态遍历性随机过程，$x_1^n$是$L$的样本，则有：
$$
H(L,q) = - \lim_{n\to\infin}\frac{1}{n}\log q(x_1^n)
$$

#### 困惑度

$$
PP_q=2^{H(L,q)}
$$

#### 互信息

$$
I(X;Y) = H(X) - H(X|Y) = H(Y)-H(Y|X)=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)}
$$

可以衡量两个字是一个词的概率。

### 人工神经网络基础

#### BP算法

误差反向传播。

输入样本、学习率，初始化权重w、偏置b，反复执行：1）正向传播信息：选定样本，算出估计。2）反向传播误差：依照估计与实际值，根据损失函数计算梯度，调整权重与偏置。

### 应用举例

#### 语义消歧

## 第三章 形式语言与自动机

### 形式语言

#### 语言描述的三种途径

穷举法、语法描述、自动机

#### 直观意义

精确描述语言及其结构的手段，以重写规则$\alpha \to \beta$的形式表示。其中$\alpha$、$\beta$均为字符串。

#### 形式语法的定义

形式语法是一个四元组$G=(N,\Sigma,P,S)$，$N$是非终结符的有限集合，$\Sigma$是终结符的有限集合，$V=N\cup\Sigma $称为总词汇表，$P$是一组重写规则的有限集合，$S\in N$，称为句子符或初始符。

如：$G=(\{A,S\},\{0,1\},P,S)$，$P:S\to 0A1,0A\to00A1,A\to1$

设$G=(N,\Sigma,P,S)$是一个文法，在$V=N\cup\Sigma $上定义关系：若$\alpha\beta\gamma$是$V=N\cup\Sigma $中的符号串，且$\beta\to\delta$是P的产生式，那么：
$$
\alpha\beta\gamma\Rightarrow_G\alpha\beta\gamma
$$

##### 最左推导

每步推导中只改写最左边的非终止符。

##### 最右推导（规范推导）

每步推导中只改写最右边的非终止符。

#### 正则文法

如果文法$G=(N,\Sigma,P,S)$的$P$中的规则满足：$A\to Bx$或$A\to x$，其中A、B

#### 上下文无关文法

$P$中的规则满足：$A\to\alpha$



### 有限自动机与正则文法

#### 确定有限自动机

$M=(\Sigma,Q,\delta,q_0,F)$

字母表$\Sigma$，状态集$Q$，转移函数$\delta$，初始状态$q_0$，终止状态$F$

#### DFA

### 下退自动机与CFG

### 有限自动机在NLP中的应用

#### 拼写检查

编辑距离

