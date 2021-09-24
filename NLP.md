# NLP

[TOC]

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

##### 穷举法

适合于句子数目有限的语言

##### 语法描述

生成语言中合格的句子、自动机

##### 自动机

对输入的句子进行检验，区别哪些是语言中的句子，哪些不是。

#### 直观意义

精确描述语言及其结构的手段，以重写规则$\alpha \to \beta$的形式表示。其中$\alpha$、$\beta$均为字符串。一个初步的字符串通过不断地运用重写规则就可以得到新的字符串。

#### 形式语法的定义

形式语法是一个四元组$G=(N,\Sigma,P,S)$，$N$是非终结符的有限集合，$\Sigma$是终结符的有限集合，$V=N\cup\Sigma $称为总词汇表，$P$是一组重写规则的有限集合，$S\in N$，称为句子符或初始符。

如：$G=(\{A,S\},\{0,1\},P,S)$，$P:S\to 0A1,0A\to00A1,A\to1$

设$G=(N,\Sigma,P,S)$是一个文法，在$(N\cup\Sigma)^* $（\*代表闭包）上定义关系：若$\alpha\beta\gamma$是$(N\cup\Sigma)^* $中的符号串，且$\beta\to\delta$是P的产生式，那么：
$$
\alpha\beta\gamma\Rightarrow_G\alpha\beta\gamma
$$

#### 推导的定义

用$\Rightarrow^+_G$（按非平凡方式派生）表示$\Rightarrow_G$的传递闭包。

用$\Rightarrow^*_G$（派生）表示$\Rightarrow_G$的自反和传递闭包。

如果清楚某个推导是文法G所产生的，则上面符号G可省略。

##### 最左推导

每步推导中只改写最左边的非终止符。

##### 最右推导（规范推导）

每步推导中只改写最右边的非终止符。

#### 句子和句型

一些特殊类型的符号串是文法$G=(N,\Sigma,P,S)$的句子形式：1）S是一个句子形式；2）如果$\alpha\beta\gamma$是一个句子形式，且$\beta\to\sigma$是$P$的产生式，则$\alpha\beta\gamma$是一个句子形式。

文法$G$的不含非终结符的句子形式被称为G生成的句子。由文法$G$生成的语言，记作$L(G)$，指$G$生成的所有句子的集合。即：$L(G)=\{x|x\in\Sigma,S\Rightarrow^+_Gx\}$

#### 正则文法

如果文法$G=(N,\Sigma,P,S)$的$P$中的规则满足：$A\to Bx$或$A\to x$，其中$A$、$B\in N,x\in\Sigma$，则称该文法为正则文法或称3型文法。

如果$A\to xB$，则该文法称为**右线性正则文法**。

#### 上下文无关文法

$P$中的规则满足：$A\to\alpha$，其中$A\in N,\alpha\in (N\cup\Sigma)^*$，则称该文法为**上下文无关文法**。

#### 上下文有关文法

$P$中的规则满足：$\alpha A\beta\to \alpha\gamma\beta$，其中$A\in N,\alpha,\beta,\gamma\in(N\cup\Sigma)^*$，且$\gamma$至少包含一个字符，则该文法为**上下文有关文法**。

#### 无约束文法（无限制重写系统）

如果$P$中的规则满足如下形式：$\alpha \to\beta$，$\alpha,\beta$是字符串，则称$G$是**无约束文法**。

### 有限自动机与正则文法

#### 确定有限自动机

$M=(\Sigma,Q,\delta,q_0,F)$

字母表$\Sigma$，状态集$Q$，转移函数$\delta \in Q\times\Sigma\to Q$，初始状态$q_0$，终止状态$F$。

##### 状态变换图

![image-20210916204703030](C:\Users\surafce book2\AppData\Roaming\Typora\typora-user-images\image-20210916204703030.png)

##### DFA定义的语言

如果一个句子$x$使得有限自动机$M$有：$\delta(q_0,x)=p,p\in F$，那么称句子$x$被$M$接受。由$M$定义的语言$T(M)$就是**被$M$接受的句子的全集**，即：
$$
T(M)=\{x|\delta(q_0,x)\in F\}
$$

举例：

![image-20210916212325423](C:\Users\surafce book2\AppData\Roaming\Typora\typora-user-images\image-20210916212325423.png)

#### 不确定有限自动机

$M=(\Sigma,Q,\delta,q_0,F)$

字母表$\Sigma$，状态集$Q$，转移函数$\delta \in Q\times\Sigma\to 2^Q$，初始状态$q_0$，终止状态$F$。

#### DFA与NFA区别与联系

##### 区别

NFA中$\delta$对应一个状态集合，而DFA中$\delta$是一个状态。

##### 关系

设L是一个被NFA接受的句子的集合，则存在一个DFA，他能接受L。

#### 正则文法与有限自动机的关系

##### 由正则文法构造DFA的步骤

令$\Sigma=V_T,Q=v_N\cup\{T\},q_0=S$，其中T是一个新增加的非终结符。

- 如果在$P$中有产生式 $S\to\epsilon$，则$F=\{S,T\}$，否则$F=\{T\}$。
- 如果在$P$中有产生式 $B\to a,B\in V_N,a\in V_T$，则$T\in\delta(B,a)$。
- 如果在$P$中有产生式 $B\to aC,B,C\in V_N,a\in V_T$，则$C\in\delta(B,a)$。
- 对于每一个$a\in V_T$，有$\delta(T,a)=\empty$。

定理：如果$G=(V_N,V_T,P,S)$是一个正则文法，则存在一个DFA $M$，使得：$T(M)=L(G)$。

##### 由DFA构造正则文法的一般步骤

令$V_N=Q,V_T=\Sigma,S=q_0$。

- 如果$C\in \delta(B,a),B,C\in Q,a\in \Sigma$，则在$P$中有产生式$B\to aC$
- 如果$C\in\delta(B,a),C\in F$，则在$P$中有产生式$B\to a$

定理：如果$M$是一个DFA，则存在一个正则文法$G=(V_N,V_T,P,S)$，使得：$L(G)=T(M)$。

### 下推自动机与CFG

### 有限自动机在NLP中的应用

#### 拼写检查

编辑距离

## 第四课 语料库

### 语料库概念

**定义** 具有既定格式与标记的大量的文本。

例如：布朗语料库

**种类** 

*共时语料库与历时语料库*

共时：研究的是一个共时平面上的元素与元素的关系。

*通用语料库与专用语料库*

*生语料与标注语料库*

### 语料库语言学

基于语料库进行语言学的研究。

### 语料库发展史

### 典型语料库

### 语料库加工方法

#### 文本处理

如垃圾格式问题。语料库内容来源复杂，存在杂质。杂质包括：文档页眉和分隔符、排版代码、表和图。如果数据来源于OCR，存在识别错误等问题，因此需要过滤器过滤这些杂质。

#### 格式标注

句通用标记语言

- SGML是超文本格式的最高层次标准，是可以定义标记语言的元语言。 
- HTML和XML同样派生于它：XML可以被认为是它的一个子集，XML的产生就是为了简化它，以便用于更加通用的目的。而HTML是它的一个应用。

#### 数据标注

语法标注

#### 搭配抽取

##### 频率方法

- 如果两个词在一起出现很多次，它们很有可能是搭配 
- 仅仅选择最频繁出现的二元组，结果并不理想

##### 均值和方差方法

- 基于频率的搜索方法可以很好的解决固定搭配的识别问题，但 是很多搭配是两词搭配，并且彼此之间的关系非常灵活

## 第五课 n元语法

### 基本概念

如何计算一段文字的概率？

如何以一段文字（句子）为单位统计相对频率？

如何根据句子构成单位的概率计算联合概率？

如句子$s=w_1w_2\dots w_m$的先验概率：
$$
p(s) = p(w_1)p(w_2|w_1)p(w_3|w_2w_1)\dots p(w_m|w_{m-1}\dots w_1)
$$
当$i=1$时，$p(w_1|w_0)=p(w_1)$

**说明**

- $w_i$可以是字、词、短语或词类等，称为统计基元。
- $w_i$的概率由$w_1,\dots,w_{i-1}$决定，有特定的一组$w_1,\dots,w_{i-1}$构成的一个序列，称为$w_i$的**历史**

但是，很长的话会导致参数很多。因此需要减少历史基元的个数。

### n元语法模型

只看最近的$n-1$个历史基元。

为了保证条件概率在$i=1$时有意义，且保证句子内所有字符串概率和为1，则在句子两端加标志**<BOS><EOS>**

不失一般性，对于$n>2$的n-gram，$p(s)$可分解为：
$$
p(s)=\prod_{i=1}^{m+1}p(w_i|w_{i-n+1}^{i-1})
$$
其中：$w_i^j$表示词序列$w_i\dots w_j$

例子：John read a book

加标记：<BOS>John read a book<EOS>

bigram：(<BOS>John),(John read) ,(read a),(a book), (book<EOS>)

### 参数估计

两个概念：

##### 训练语料

用于建立模型，确定模型参数的已知语料。

##### 最大似然估计

用相对概率计算概率的方法。

对于n-gram，参数$p(w_i|w^{i-1}_{i-n+1})$可由最大似然估计求得：
$$
p(w_i|w^{i-1}_{i-n+1})=f(w_i|w^{i-1}_{i-n+1})=\frac{c^i_{i-n+1}}{\sum_{w_i}c(w^i_{i-n+1})}
$$
式中：$\sum_{w_i}c(w^i_{i-n+1})$是历史串$w^{i-1}_{i-n+1}$在给定语料中出现的次数。$f(w_i|w^{i-1}_{i-n+1})$是在给定$w^{i-1}_{i-n+1}$的条件下$w_i$出现的相对频度。$c^i_{i-n+1}$是$w^{i-1}_{i-n+1}$与$w_i$同时出现的次数。

举个例子：

- John read Moby Dick
- Mary read a different book
- She read a book by Cher

根据二元语法求“John read a book”句子的概率。
$$
p(John|<BOS>)=\frac{c(<BOS>John)}{\sum_wc(<BOS>w)}=\frac 1 3
$$

$$
p(read|John)=\frac{c(John\ read)}{\sum_wc(John\ w)}=\frac 1 1
$$

$$
p(a|read)=\frac{c(read\ a)}{\sum_wc(read\ w)}=\frac 2 3
$$

$$
p(book|a)=\frac{c(a\ book)}{\sum_wc(a\ w)}=\frac 1 2
$$

$$
p(<EOS>|book)=\frac{c(book<EOS>)}{\sum_wc(book\ w)}=\frac 1 2
$$

所以：
$$
p(John\ read\ a\ book)=\frac 1 3 \times1\times \frac 2 3\times \frac 1 2\times\frac 1 2=0.06
$$
但是，数据稀疏会引起零概率问题，需要引入数据平滑。

### 数据平滑

#### 基本思想

消除零概率，改进模型整体正确率

#### 基本目标

测试样本的语言模型[困惑度](# 困惑度)越小越好

#### 基本约束

$$
\sum_{w_i}p(w_i|w_1,w_2,\dots,w_{i-1})=1
$$

#### 平滑方法

##### 加一法

每一种情况都加一。对于2元语法有：
$$
p(w_i|w_{i-1})=\frac{1+c(w_{i-1}w_i)}{\sum_{w_i}(1+c(w_{i-1}w_i))}=\frac{1+c(w_{i-1}w_i)}{|V|+\sum_{w_i}c(w_{i-1}w_i)}
$$
