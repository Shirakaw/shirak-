

### 使用PyTorch构建神经网络以及反向传播计算

> 前一段时间南京出现了疫情，大概原因是因为境外飞机清洁处理不恰当，导致清理人员感染。话说国外一天不消停，国内就得一直严防死守。沈阳出现了一例感染人员，我在22号乘坐飞机从沈阳乘坐飞机到杭州，恰好我是一位密切接触人员的后三排，就这样我成为了次密切接触人员，人下飞机刚到杭州就被疾控中心带走了，享受了全免费的隔离套餐，不得不说疾控中心大数据把控是真的有力度。在这一段时间，也让我沉下心来去做了点事，之前一直鸽的公众号也开始写上了。。。不过隔离期间确实让我这么宅的人都感觉很憋。。。唠唠叨叨这么多，言归正传，这节将从反向传播算法进一步分析基于PyTorch的神经网络算法。

### 构建神经网络

在训练神经网络时，常用的算法是反向传播。该算法中，参数根据损失函数相对应的给定的参数的梯度进行调整。为了计算这些梯度，使用PyTorch的内置函数**torch.autograd**。它支持任何网络的梯度计算。
通过构建一层神经网络来进行细致的分析；
  ```python
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
  ```
通过PyTorch定义具有输入**x**，参数**w**,**b**以及损失函数，进而构建一层神经网络。

### 张量，函数，计算图

![](https://cdn.jsdelivr.net/gh/filess/img9@main/2021/08/01/1627828479610-868e5c39-2628-4a07-9f24-906cc49500b3.png)

  ```python
print('Gradient function for z =',z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)
  ```
  输出结果如下；
![](https://cdn.jsdelivr.net/gh/filess/img8@main/2021/08/01/1627829108084-7ac6cc9b-07b7-4002-a574-0e640e1973fb.png)
我们通过张量来构建计算图函数本质是**Function**的对象。该对象定义了前向计算函数，在反向传播中计算导数。将反向传播函数存储在**grad_fn**中。

  >参考PyTorch文档https://pytorch.org
在整个网络中，**w**，**b**是需要优化的参数。为了计算这些参数的损失函数的梯度，设置了**requires_grad**这些张量的属性。

### 计算梯度

梯度，本意是一个向量，表示某一函数在该点处的方向导数沿着该方向取得最大值，即函数在该点处沿着该方向(此梯度的方向)变化最快，变化率最大。
为了优化神经网络参数的权重，需要计算损失函数关于参数的导数，通过**loss.backward()** 进行计算。

  ```python
loss.backward()
print(w.grad)
print(b.grad)
  ```
输出结果如下；
![](https://cdn.jsdelivr.net/gh/filess/img19@main/2021/08/02/1627865480328-d3fccbc7-7710-4691-8a36-f0a2c6822c0c.png)
>我们只能获取**grad**计算图的叶子节点的**requeires_grad**属性，这些属性设置为True。**backward**只能在图中使用一次梯度计算，如果需要**backward**在同一个图多次调用，我们需要传递**retain_graph-True**和**backward**调用

### 更多图计算
从概念上讲，**autograd** 在由**Function** 对象组成的有向无环图 (DAG) 中保存数据（张量）和所有已执行操作（以及生成的新张量）的记录 。在这个 **DAG** 中，叶子是输入张量，根是输出张量。通过从根到叶跟踪此图，可以使用链式法则自动计算梯度。
在前向传递过程，**autograd**需要计算结果张量，在**DAG**中保持梯度函数
在**DAG**中调用**abckward()** 需要计算各自的梯度，将其累计在各自的**grad**属性中，使用链式法则，传递到叶张量。
>**DAG**是动态的，每次调用**backward()**函数，**autograd**开始填充新图形，如果需要可以在每次迭代时进行操作。
 ### 梯度以及雅可比积
 梯度，张量在任意常数c方向上的梯度：**n阶**张量的梯度是**n+1**张量场。
 雅可比积是表示两个向量的所有可能偏导数的矩阵。它是一个向量相对于另一个向量的梯度，**Autograd**可以对张量进行微分，从一个变量开始执行反向传播。在深度学习中，这个变量通常保存成本函数的值，自动计算所有反向传播梯度
![](https://cdn.jsdelivr.net/gh/filess/img13@main/2021/08/02/1627896366645-4bcf63e2-9261-4b26-893c-b5b8d4f74dc4.png)
计算雅可比乘积而不是计算雅可比矩阵本身；

计算成绩如下所示
   ```python
inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nSecond call\n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)
 ```

![](https://cdn.jsdelivr.net/gh/filess/img11@main/2021/08/02/1627896942310-cb104b62-af3d-40b0-a13d-0b1a058dbdb8.png)

当**backward**使用相同的参数进行第二次调用时，梯度的值是不同的。发生这种情况是因为在进行反向传播时，**PyTorch**会积累梯度，即将梯度的值添加到**grad**图计算中所有节点的属性中。如果要计算适当的梯度，需要将**grad**归零。

### 结语

关于禁用渐变跟踪问题上，有时我们不需要跟踪计算所有计算历史，将神经网络某些参数固定是微调神经网络常用方法，而且在仅向前传递时加快计算速度。
在图计算问题上，autograd 在由Function 对象组成的有向无环图 (DAG) 中保存数据（张量）和所有已执行操作（以及生成的新张量）的记录 。在这个 DAG 中，叶子是输入张量，根是输出张量。通过从根到叶跟踪此图，您可以使用链式法则自动计算梯度。前向传递，autograd计算结果张量，在dag中维护操作的梯度函数，在反向传递中，计算每个梯度.grad_fn，将他们积累到.grad属性中，使用链式法则传播到叶张量。下一节将从优化模型参数角度进行学习
#### 推荐阅读

- [微分算子法](https://mp.weixin.qq.com/s/yz3x4JtgnC0lSSOLrBp5lA)
- [使用PyTorch构建神经网络模型进行手写识别](https://mp.weixin.qq.com/s/TPPYYOxRWuQMLEH9Mkeo-g)
- [使用PyTorch构建神经网络模型以及反向传播计算](https://mp.weixin.qq.com/s/aOGm3rQuA6ASBPPt8IiivA)
![](https://cdn.jsdelivr.net/gh/filess/img1@main/2021/07/31/1627739160091-8a709507-cda2-476a-9b44-8a35ff0212e4.jpg)

