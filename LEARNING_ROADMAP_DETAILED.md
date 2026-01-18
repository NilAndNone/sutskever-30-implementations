# 从零吃透深度学习原理 - 详细路线图

> 适用于：有大模型使用经验，但不懂底层原理的学习者

---

## 前置知识自测

在开始前，确认你能做这些（不能的话先补）：

```python
# 1. NumPy 基础 - 你能看懂这些吗？
a = np.array([[1,2],[3,4]])
b = a @ a.T                    # 矩阵乘法
c = a.reshape(-1)              # 展平
d = np.sum(a, axis=1)          # 按行求和

# 2. 求导 - 你能算这些吗？
f(x) = x²        → f'(x) = ?   # 答案：2x
f(x) = e^x       → f'(x) = ?   # 答案：e^x
f(x) = 1/(1+e^-x) → f'(x) = ?  # 答案：f(x)(1-f(x))

# 3. 链式法则
z = f(g(x))      → dz/dx = ?   # 答案：f'(g(x)) · g'(x)
```

**如果上面有不会的**：先花 2-3 天补 NumPy 和微积分基础。

---

## 阶段 0：神经网络最小单元（3-5天）

### 目标
理解单个神经元是怎么工作的。

### 学习内容

#### 0.1 线性变换
```
输入 x → 乘权重 w → 加偏置 b → 输出 y = wx + b
```

**习题 0.1**：
```python
# 手算：x=2, w=3, b=1，求 y
# 答案：y = 3*2 + 1 = 7
```

#### 0.2 激活函数
```
为什么需要？→ 没有激活函数，多层网络等于单层（线性叠加还是线性）
```

**三个必须掌握的激活函数**：
```python
# Sigmoid: 把任意数压到 (0,1)
sigmoid(x) = 1 / (1 + exp(-x))

# ReLU: 负数变0，正数不变
relu(x) = max(0, x)

# Tanh: 把任意数压到 (-1,1)
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

**习题 0.2**：
```python
# 1. 手算 sigmoid(0) = ?     答案：0.5
# 2. 手算 relu(-3) = ?       答案：0
# 3. 手算 relu(5) = ?        答案：5
# 4. 为什么 ReLU 比 Sigmoid 更常用？
#    答案：计算简单 + 不容易梯度消失
```

#### 0.3 单个神经元完整流程
```
输入 x → 线性变换 z=wx+b → 激活函数 a=σ(z) → 输出
```

**习题 0.3**：
```python
# 实现单个神经元
def neuron(x, w, b):
    z = w * x + b
    a = 1 / (1 + np.exp(-z))  # sigmoid
    return a

# 测试：x=1, w=2, b=-1
# z = 2*1 + (-1) = 1
# a = sigmoid(1) ≈ 0.731
```

#### 0.4 多输入神经元
```
x = [x1, x2, x3]
w = [w1, w2, w3]
z = w1*x1 + w2*x2 + w3*x3 + b = w·x + b（点积）
```

**习题 0.4**：
```python
# 手算：x=[1,2,3], w=[0.5,0.5,0.5], b=0
# z = 0.5*1 + 0.5*2 + 0.5*3 + 0 = 3
# a = sigmoid(3) ≈ 0.953

# 代码实现
def neuron_multi(x, w, b):
    z = np.dot(w, x) + b
    return sigmoid(z)
```

### 阶段 0 检验
- [ ] 能手算单个神经元的输出
- [ ] 能解释为什么需要激活函数
- [ ] 能写出 sigmoid/relu 的公式和代码

---

## 阶段 1：前向传播（5-7天）

### 目标
理解数据如何从输入流向输出。

### 学习内容

#### 1.1 两层网络
```
输入层 (3个神经元) → 隐藏层 (4个神经元) → 输出层 (2个神经元)

参数：
W1: 4×3 矩阵（隐藏层权重）
b1: 4×1 向量（隐藏层偏置）
W2: 2×4 矩阵（输出层权重）
b2: 2×1 向量（输出层偏置）
```

**习题 1.1**：
```python
# 1. 如果输入是 5 维，隐藏层 10 个神经元，输出 3 维
#    W1 的 shape 是？ 答案：(10, 5)
#    W2 的 shape 是？ 答案：(3, 10)

# 2. 总共有多少个参数？
#    W1: 10*5=50, b1: 10, W2: 3*10=30, b2: 3
#    总计：50+10+30+3 = 93 个参数
```

#### 1.2 前向传播代码
```python
def forward(x, W1, b1, W2, b2):
    # 第一层
    z1 = W1 @ x + b1      # 线性变换
    a1 = relu(z1)         # 激活

    # 第二层
    z2 = W2 @ a1 + b2     # 线性变换
    a2 = softmax(z2)      # 输出层用 softmax（分类任务）

    return a2
```

**习题 1.2**：
```python
# 手动追踪维度
# 输入 x: (3,)
# W1: (4,3), b1: (4,)
# z1 = W1 @ x + b1 → shape: (4,)
# a1 = relu(z1) → shape: (4,)
# W2: (2,4), b2: (2,)
# z2 = W2 @ a1 + b2 → shape: (2,)
# a2 = softmax(z2) → shape: (2,)
```

#### 1.3 Softmax 函数
```python
# 把任意数变成概率分布（和为1）
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # 减最大值防溢出
    return exp_z / np.sum(exp_z)

# 例：z = [2, 1, 0]
# exp_z = [7.39, 2.72, 1]
# sum = 11.11
# softmax = [0.67, 0.24, 0.09]  # 和为 1
```

**习题 1.3**：
```python
# 1. softmax([0, 0, 0]) = ?  答案：[1/3, 1/3, 1/3]
# 2. softmax([100, 0, 0]) ≈ ? 答案：[1, 0, 0]（几乎全给最大的）
# 3. 为什么分类任务用 softmax？
#    答案：输出是概率，可以表示"属于每个类的可能性"
```

#### 1.4 损失函数
```python
# 交叉熵损失（分类任务最常用）
def cross_entropy_loss(y_pred, y_true):
    # y_true: one-hot 编码，如 [0, 1, 0] 表示第 2 类
    # y_pred: softmax 输出，如 [0.1, 0.7, 0.2]
    return -np.sum(y_true * np.log(y_pred + 1e-8))

# 例：预测 [0.1, 0.7, 0.2]，真实标签是第 2 类
# loss = -log(0.7) ≈ 0.357
```

**习题 1.4**：
```python
# 1. 预测 [0.9, 0.1]，真实是第 1 类 [1, 0]
#    loss = -log(0.9) ≈ 0.105（小，说明预测对了）

# 2. 预测 [0.1, 0.9]，真实是第 1 类 [1, 0]
#    loss = -log(0.1) ≈ 2.303（大，说明预测错了）

# 3. 为什么用 log？
#    答案：预测越接近 1，log 越接近 0（惩罚小）
#          预测越接近 0，log 越接近 -∞（惩罚大）
```

### 阶段 1 检验
打开 `02_char_rnn_karpathy.ipynb`，只看前向传播部分，确保你能：
- [ ] 追踪每一层的数据 shape
- [ ] 解释每一步在做什么
- [ ] 手算一个简单的两层网络

---

## 阶段 2：反向传播（7-10天）

### 目标
理解网络如何学习（梯度如何计算和传递）。

### 核心思想
```
损失 L 对每个参数 w 求导 → 得到梯度 dL/dw → 更新 w = w - lr * dL/dw
```

### 学习内容

#### 2.1 链式法则复习
```
如果 y = f(g(x))，则 dy/dx = dy/dg · dg/dx

例：y = (2x + 1)²
设 g = 2x + 1，则 y = g²
dy/dg = 2g
dg/dx = 2
dy/dx = 2g · 2 = 4(2x + 1)
```

**习题 2.1**：
```python
# 1. y = sigmoid(3x + 2)，求 dy/dx
#    设 z = 3x + 2
#    dy/dz = sigmoid(z)(1-sigmoid(z))
#    dz/dx = 3
#    dy/dx = 3 · sigmoid(z)(1-sigmoid(z))

# 2. y = relu(wx + b)，求 dy/dw
#    设 z = wx + b
#    dy/dz = 1 if z > 0 else 0
#    dz/dw = x
#    dy/dw = x if z > 0 else 0
```

#### 2.2 单层网络的反向传播
```
前向：z = wx + b → a = σ(z) → L = loss(a, y)

反向（从后往前）：
dL/da = ... （损失对输出的导数）
da/dz = σ'(z)（激活函数的导数）
dz/dw = x（线性层对权重的导数）
dz/db = 1（线性层对偏置的导数）

链式法则：
dL/dw = dL/da · da/dz · dz/dw
dL/db = dL/da · da/dz · dz/db
```

**习题 2.2**：
```python
# 假设：
# 输入 x = 2
# 权重 w = 0.5, b = 0
# 激活函数 sigmoid
# 真实标签 y = 1
# 损失函数 MSE: L = (a - y)²

# 前向传播：
z = 0.5 * 2 + 0 = 1
a = sigmoid(1) ≈ 0.731
L = (0.731 - 1)² ≈ 0.072

# 反向传播：
dL_da = 2(a - y) = 2(0.731 - 1) = -0.538
da_dz = a(1-a) = 0.731 * 0.269 ≈ 0.197
dz_dw = x = 2

dL_dw = dL_da * da_dz * dz_dw
      = -0.538 * 0.197 * 2 ≈ -0.212

# 更新（学习率 lr=0.1）：
w_new = w - lr * dL_dw = 0.5 - 0.1*(-0.212) = 0.521
```

#### 2.3 两层网络的反向传播
```
前向：
x → [W1, b1] → z1 → relu → a1 → [W2, b2] → z2 → softmax → a2 → loss

反向（关键是正确传递梯度）：
dL/da2 → dL/dz2 → dL/dW2, dL/db2
                → dL/da1 → dL/dz1 → dL/dW1, dL/db1
```

**习题 2.3**：
```python
# 完成反向传播代码的填空
def backward(x, y_true, W1, b1, W2, b2, cache):
    a1, z1, a2, z2 = cache
    m = x.shape[0]  # batch size

    # 输出层梯度
    dz2 = a2 - y_true                    # softmax + cross entropy 的导数
    dW2 = (1/m) * dz2 @ a1.T             # ___填空___
    db2 = (1/m) * np.sum(dz2, axis=1)    # ___填空___

    # 传到隐藏层
    da1 = W2.T @ dz2                     # ___填空___
    dz1 = da1 * (z1 > 0)                 # relu 的导数
    dW1 = (1/m) * dz1 @ x.T              # ___填空___
    db1 = (1/m) * np.sum(dz1, axis=1)    # ___填空___

    return dW1, db1, dW2, db2
```

#### 2.4 梯度检验
```python
# 数值梯度：用定义验证你的解析梯度是否正确
def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x[i] += eps
        f_plus = f(x)
        x[i] -= 2*eps
        f_minus = f(x)
        x[i] += eps
        grad[i] = (f_plus - f_minus) / (2*eps)
    return grad

# 如果 |解析梯度 - 数值梯度| / |解析梯度 + 数值梯度| < 1e-5
# 说明你的反向传播写对了
```

**习题 2.4**：
```python
# 对你写的反向传播做梯度检验
# 如果误差 > 1e-4，说明代码有 bug
```

### 阶段 2 检验
打开 `02_char_rnn_karpathy.ipynb`，找到反向传播部分：
- [ ] 能追踪梯度如何从 loss 流回第一层
- [ ] 能手算一个简单网络的梯度
- [ ] 理解为什么叫"反向"传播

---

## 阶段 3：RNN 与序列模型（7-10天）

### 目标
理解如何处理序列数据（文本、时间序列）。

### 核心问题
```
普通神经网络：输入固定长度，输出固定长度
序列问题：输入/输出长度可变，且有先后顺序关系

例：翻译 "我爱你" → "I love you"
    输入 3 个字，输出 3 个词，且顺序有关系
```

### 学习内容

#### 3.1 RNN 基本结构
```
时间步 t=0: x0 → [RNN] → h0
时间步 t=1: x1, h0 → [RNN] → h1
时间步 t=2: x2, h1 → [RNN] → h2
...

关键：每一步都用同样的参数，但会接收上一步的隐藏状态
```

```python
# RNN 单步
def rnn_step(x_t, h_prev, Wxh, Whh, bh):
    h_t = np.tanh(Wxh @ x_t + Whh @ h_prev + bh)
    return h_t

# 完整序列
def rnn_forward(xs, h0, Wxh, Whh, bh):
    h = h0
    hs = []
    for x_t in xs:
        h = rnn_step(x_t, h, Wxh, Whh, bh)
        hs.append(h)
    return hs
```

**习题 3.1**：
```python
# 1. 如果输入维度是 10，隐藏层维度是 20
#    Wxh 的 shape 是？ 答案：(20, 10)
#    Whh 的 shape 是？ 答案：(20, 20)

# 2. 为什么 Whh 是方阵？
#    答案：h_prev 和 h_t 维度相同，都是隐藏层维度

# 3. RNN 和普通网络的核心区别是什么？
#    答案：RNN 有"记忆"（通过 h 传递历史信息）
```

#### 3.2 BPTT（时间反向传播）
```
普通反向传播：一层一层往回传
BPTT：一个时间步一个时间步往回传

关键：梯度要沿着时间流回去
dL/dh2 → dL/dh1 → dL/dh0

问题：序列很长时，梯度会指数衰减（梯度消失）
```

**习题 3.2**：
```python
# 假设序列长度为 100，每一步梯度乘以 0.9
# 从最后传到第一步，梯度变成多少？
# 答案：0.9^100 ≈ 0.000027（几乎为 0）

# 这就是为什么普通 RNN 难以学习长距离依赖
```

#### 3.3 LSTM 解决梯度消失
```
核心思想：添加一条"高速公路"让梯度直接流过

RNN:  h_t = tanh(Wx·x_t + Wh·h_{t-1})
LSTM: c_t = f_t * c_{t-1} + i_t * g_t  ← 关键！这里是加法，不是乘法

加法的好处：梯度直接流过，不会衰减
```

**LSTM 四个门**：
```python
# 遗忘门：决定丢弃多少旧信息
f_t = sigmoid(Wf @ [h_{t-1}, x_t] + bf)

# 输入门：决定添加多少新信息
i_t = sigmoid(Wi @ [h_{t-1}, x_t] + bi)

# 候选记忆：新的候选信息
g_t = tanh(Wg @ [h_{t-1}, x_t] + bg)

# 输出门：决定输出多少信息
o_t = sigmoid(Wo @ [h_{t-1}, x_t] + bo)

# 更新
c_t = f_t * c_{t-1} + i_t * g_t  # cell state
h_t = o_t * tanh(c_t)            # hidden state
```

**习题 3.3**：
```python
# 1. 如果 f_t = 1, i_t = 0，cell state 会怎样？
#    答案：c_t = c_{t-1}，完全保留旧信息

# 2. 如果 f_t = 0, i_t = 1，cell state 会怎样？
#    答案：c_t = g_t，完全用新信息替换

# 3. 为什么门用 sigmoid？
#    答案：sigmoid 输出 0-1，正好表示"保留多少比例"

# 4. 画出 LSTM 的数据流图（这是最重要的习题！）
```

### 阶段 3 核心作业

**作业 3A**：打开 `02_char_rnn_karpathy.ipynb`
- [ ] 完整运行一遍
- [ ] 理解字符级语言模型的输入输出
- [ ] 修改温度参数，观察生成文本的变化

**作业 3B**：打开 `03_lstm_understanding.ipynb`
- [ ] 画出 LSTM 的完整结构图（纸上）
- [ ] 写出四个门的公式（不看代码）
- [ ] 运行 notebook，观察门的激活值可视化

### 阶段 3 检验
- [ ] 能解释 RNN 为什么会梯度消失
- [ ] 能画出 LSTM 结构，写出公式
- [ ] 能说出 LSTM 如何解决梯度消失

---

## 阶段 4：注意力机制（10-14天）

### 目标
理解现代 AI 的核心机制。

### 核心问题
```
RNN/LSTM 的瓶颈：
- 信息必须通过 hidden state 传递
- 长序列后，早期信息会丢失
- 无法并行计算

注意力的解决方案：
- 直接建立输入和输出的连接
- 想看哪里就看哪里，不用按顺序
- 可以并行计算
```

### 学习内容

#### 4.1 注意力的直觉
```
翻译 "我爱你" → "I love you"

翻译 "I" 时，应该主要看"我"
翻译 "love" 时，应该主要看"爱"
翻译 "you" 时，应该主要看"你"

注意力 = 决定"看哪里"的机制
```

#### 4.2 注意力计算（三步）
```
Query (Q): 我想找什么？
Key (K): 每个位置的"标签"
Value (V): 每个位置的"内容"

步骤：
1. 计算相似度：score = Q · K^T
2. 归一化：weights = softmax(score)
3. 加权求和：output = weights · V
```

```python
def attention(Q, K, V):
    # Q: (seq_len_q, d_k)
    # K: (seq_len_k, d_k)
    # V: (seq_len_k, d_v)

    scores = Q @ K.T           # (seq_len_q, seq_len_k)
    weights = softmax(scores)  # (seq_len_q, seq_len_k)
    output = weights @ V       # (seq_len_q, d_v)
    return output, weights
```

**习题 4.1**：
```python
# Q = [[1, 0]]  (1个query)
# K = [[1, 0], [0, 1], [1, 1]]  (3个key)
# V = [[1], [2], [3]]  (3个value)

# 1. 计算 scores = Q @ K.T
#    答案：[[1, 0, 1]]

# 2. 计算 weights = softmax(scores)
#    答案：softmax([1, 0, 1]) ≈ [0.42, 0.16, 0.42]

# 3. 计算 output = weights @ V
#    答案：0.42*1 + 0.16*2 + 0.42*3 ≈ 2.0

# 4. 解释：为什么 query [1,0] 更关注 key [1,0] 和 [1,1]？
#    答案：因为点积更大，说明方向更相似
```

#### 4.3 Scaled Dot-Product Attention
```python
def scaled_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)  # 除以 sqrt(d_k)
    weights = softmax(scores)
    return weights @ V
```

**习题 4.2**：
```python
# 为什么要除以 sqrt(d_k)？

# 假设 Q, K 的每个元素都是均值 0、方差 1 的随机数
# Q · K 的方差 = d_k（维度越高，方差越大）
# 如果不除，scores 会很大 → softmax 会饱和 → 梯度消失

# 例：d_k = 64
# 不除：scores 的标准差 ≈ 8，softmax([8, 0, 0]) ≈ [1, 0, 0]（过于尖锐）
# 除以 8：scores 的标准差 ≈ 1，softmax([1, 0, 0]) ≈ [0.58, 0.21, 0.21]（更平滑）
```

#### 4.4 Multi-Head Attention
```
为什么多头？
- 单头只能学一种"关注方式"
- 多头可以学多种（如：一个头关注语法，一个头关注语义）
```

```python
def multi_head_attention(Q, K, V, num_heads):
    d_model = Q.shape[-1]
    d_k = d_model // num_heads

    heads = []
    for i in range(num_heads):
        # 每个头有自己的投影矩阵
        Q_i = Q @ W_Q[i]  # (seq, d_k)
        K_i = K @ W_K[i]
        V_i = V @ W_V[i]
        head_i = scaled_attention(Q_i, K_i, V_i)
        heads.append(head_i)

    # 拼接所有头
    concat = np.concatenate(heads, axis=-1)  # (seq, d_model)
    output = concat @ W_O  # 最终投影
    return output
```

**习题 4.3**：
```python
# d_model = 512, num_heads = 8
# 1. 每个头的维度 d_k = ?  答案：64
# 2. 总参数量（只算 W_Q, W_K, W_V, W_O）= ?
#    W_Q: 8 * 512 * 64 = 262144
#    W_K: 同上
#    W_V: 同上
#    W_O: 512 * 512 = 262144
#    总计：262144 * 4 = 1,048,576 ≈ 1M 参数
```

#### 4.5 Self-Attention
```
Q = K = V = 同一个序列

例："我 爱 你"
每个词都和其他词计算注意力
"我" 看 "我"、"爱"、"你" 的相似度
"爱" 看 "我"、"爱"、"你" 的相似度
...
```

**习题 4.4**：
```python
# Self-attention 的输出 shape 和输入相同吗？
# 输入：(batch, seq_len, d_model)
# 输出：(batch, seq_len, d_model)
# 答案：相同！每个位置都输出一个向量
```

### 阶段 4 核心作业

**作业 4A**：打开 `14_bahdanau_attention.ipynb`
- [ ] 理解 encoder-decoder 结构
- [ ] 可视化注意力权重，理解翻译时"看哪里"

**作业 4B**：打开 `13_attention_is_all_you_need.ipynb`（最重要！）
- [ ] 写出 Scaled Dot-Product Attention 的公式
- [ ] 解释为什么除以 sqrt(d_k)
- [ ] 解释 Multi-Head 的作用
- [ ] 画出 Transformer encoder 的结构

### 阶段 4 检验
- [ ] 能手写 attention 函数
- [ ] 能解释 Q, K, V 分别代表什么
- [ ] 能画出 Transformer 的完整结构
- [ ] 能解释位置编码为什么必要

---

## 阶段 5：Transformer 深入（10-14天）

### 目标
完全理解 Transformer 架构。

### 学习内容

#### 5.1 位置编码
```
问题：Self-attention 是排列不变的
     即 attention("我爱你") = attention("你爱我")（顺序打乱结果一样）

解决：给每个位置加上位置信息
```

```python
def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / 10000**(i/d_model))
            PE[pos, i+1] = np.cos(pos / 10000**(i/d_model))
    return PE

# 输入 = 词嵌入 + 位置编码
x = embedding(tokens) + positional_encoding(seq_len, d_model)
```

**习题 5.1**：
```python
# 1. 为什么用 sin/cos 而不是简单的 [1,2,3,4...]？
#    答案：sin/cos 有周期性，可以泛化到训练时没见过的长度

# 2. 为什么不同维度用不同频率？
#    答案：低频捕捉长距离关系，高频捕捉短距离关系
```

#### 5.2 残差连接和 Layer Normalization
```python
# Transformer 的每个子层都这样包装：
def sublayer(x, layer):
    return layer_norm(x + layer(x))

# 为什么？
# 1. 残差连接：让梯度直接流过，训练更深网络
# 2. Layer Norm：稳定训练，加速收敛
```

#### 5.3 Feed-Forward Network
```python
def ffn(x, W1, b1, W2, b2):
    # 两层全连接，中间用 ReLU
    return W2 @ relu(W1 @ x + b1) + b2

# 典型配置：
# d_model = 512
# d_ff = 2048（中间层是 4 倍）
```

**习题 5.2**：
```python
# FFN 的参数量 = ?
# W1: d_model * d_ff = 512 * 2048 = 1M
# W2: d_ff * d_model = 2048 * 512 = 1M
# 总计约 2M 参数（每层）
```

#### 5.4 完整 Transformer Encoder
```python
def transformer_encoder_layer(x, W_Q, W_K, W_V, W_O, W1, b1, W2, b2):
    # Self-Attention + 残差 + LayerNorm
    attn_output = multi_head_attention(x, x, x, W_Q, W_K, W_V, W_O)
    x = layer_norm(x + attn_output)

    # FFN + 残差 + LayerNorm
    ffn_output = ffn(x, W1, b1, W2, b2)
    x = layer_norm(x + ffn_output)

    return x
```

#### 5.5 Decoder 和 Masked Attention
```
Decoder 的特殊之处：
1. Masked Self-Attention：只能看到之前的位置（防止作弊）
2. Cross-Attention：Query 来自 decoder，Key/Value 来自 encoder
```

```python
def masked_attention(Q, K, V, mask):
    scores = Q @ K.T / np.sqrt(d_k)
    scores = scores + mask  # mask 是 -inf 的矩阵
    weights = softmax(scores)  # -inf 变成 0
    return weights @ V
```

**习题 5.3**：
```python
# mask 长什么样？（seq_len = 4）
# [[0, -inf, -inf, -inf],
#  [0,    0, -inf, -inf],
#  [0,    0,    0, -inf],
#  [0,    0,    0,    0]]
#
# 第 i 行只有前 i 个位置是 0（可以看到）
```

### 阶段 5 核心作业

**作业 5A**：完整实现 Transformer encoder
```python
# 不看任何参考，自己写出：
# 1. scaled_dot_product_attention
# 2. multi_head_attention
# 3. positional_encoding
# 4. transformer_encoder_layer
# 5. 组合成完整的 encoder
```

**作业 5B**：参数量计算
```python
# 假设 BERT-base 配置：
# d_model = 768, num_heads = 12, d_ff = 3072, num_layers = 12
# 词表大小 = 30000

# 计算：
# 1. 词嵌入参数量 = ?
# 2. 单层 encoder 参数量 = ?
# 3. 总参数量 = ?
```

### 阶段 5 检验
- [ ] 能从零实现 Transformer encoder（不看代码）
- [ ] 能画出完整的 encoder-decoder 结构
- [ ] 能计算给定配置的参数量
- [ ] 能解释为什么 Transformer 能并行而 RNN 不能

---

## 阶段 6：CNN 与视觉模型（7-10天）

### 目标
理解卷积神经网络如何处理图像。

### 学习内容

#### 6.1 卷积操作
```
为什么不用全连接处理图像？
- 图片 224×224×3 = 150528 维
- 第一层 1000 个神经元 → 1.5 亿参数（太多了！）

卷积的解决方案：
- 局部连接：每个神经元只看图像的一小块
- 参数共享：同一个卷积核在整个图像上滑动
```

```python
def conv2d(x, kernel):
    """
    x: (H, W) 输入图像
    kernel: (kH, kW) 卷积核
    """
    H, W = x.shape
    kH, kW = kernel.shape
    out_H = H - kH + 1
    out_W = W - kW + 1
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            # 取出局部区域，和卷积核做元素乘法再求和
            patch = x[i:i+kH, j:j+kW]
            output[i, j] = np.sum(patch * kernel)

    return output
```

**习题 6.1**：
```python
# 输入图像 5×5，卷积核 3×3，stride=1，no padding
# 输出尺寸 = ?
# 答案：(5-3+1) × (5-3+1) = 3×3

# 输入图像 32×32，卷积核 5×5，stride=1，padding=2
# 输出尺寸 = ?
# 答案：(32+2*2-5+1) × ... = 32×32（same padding）

# 公式：out = (in + 2*padding - kernel) / stride + 1
```

#### 6.2 池化（Pooling）
```
作用：
1. 减小尺寸，减少计算量
2. 增加感受野
3. 一定程度的平移不变性
```

```python
def max_pool2d(x, size=2):
    H, W = x.shape
    out_H, out_W = H // size, W // size
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            patch = x[i*size:(i+1)*size, j*size:(j+1)*size]
            output[i, j] = np.max(patch)

    return output
```

#### 6.3 完整 CNN 结构
```
输入 (224, 224, 3)
    ↓ Conv 3×3, 64 filters
(224, 224, 64)
    ↓ MaxPool 2×2
(112, 112, 64)
    ↓ Conv 3×3, 128 filters
(112, 112, 128)
    ↓ MaxPool 2×2
(56, 56, 128)
    ... (更多层)
    ↓ Flatten
(features,)
    ↓ Fully Connected
(num_classes,)
```

**习题 6.2**：
```python
# AlexNet 第一层：
# 输入: 227×227×3
# 卷积核: 11×11×3, 96个, stride=4
#
# 1. 输出尺寸 = ?
#    (227-11)/4 + 1 = 55, 所以是 55×55×96
#
# 2. 参数量 = ?
#    每个核: 11×11×3 = 363
#    96个核: 363×96 = 34848
#    加偏置: 34848 + 96 = 34944
```

### 阶段 6 核心作业

**作业 6A**：打开 `07_alexnet_cnn.ipynb`
- [ ] 理解每一层的输入输出尺寸
- [ ] 计算每一层的参数量

**作业 6B**：打开 `26_cs231n_cnn_fundamentals.ipynb`（重点！）
- [ ] 完成从 kNN → 线性分类器 → 神经网络 → CNN 的完整流程
- [ ] 理解每种方法的优缺点

**作业 6C**：打开 `10_resnet_deep_residual.ipynb`
- [ ] 理解残差连接如何解决深度网络训练问题
- [ ] 画出残差块结构

### 阶段 6 检验
- [ ] 能手算卷积输出尺寸
- [ ] 能计算 CNN 的参数量
- [ ] 能解释为什么深度网络需要残差连接
- [ ] 能画出 AlexNet/VGG/ResNet 的基本结构

---

## 阶段 7：高级专题（每个 3-5 天）

完成前 6 个阶段后，根据兴趣选择：

### 7A: 外部记忆（NTM）
- 打开 `20_neural_turing_machine.ipynb`
- 理解 content-based 和 location-based 寻址
- 理解可微分的读写操作

### 7B: 图神经网络
- 打开 `12_graph_neural_networks.ipynb`
- 理解消息传递框架
- 能处理非欧几里得数据

### 7C: 生成模型（VAE）
- 打开 `17_variational_autoencoder.ipynb`
- 理解 ELBO 和 reparameterization trick
- 理解隐变量模型

### 7D: Scaling Laws
- 打开 `22_scaling_laws.ipynb`
- 理解参数量、数据量、计算量的关系
- 理解为什么大模型更强

### 7E: RAG
- 打开 `28_dense_passage_retrieval.ipynb` 和 `29_rag.ipynb`
- 理解检索增强生成
- 理解为什么 RAG 能减少幻觉

### 7F: 理论基础
- 打开 `23_mdl_principle.ipynb`（MDL）
- 打开 `25_kolmogorov_complexity.ipynb`（复杂度）
- 打开 `24_machine_super_intelligence.ipynb`（AIXI）

---

## 总时间估计

| 阶段 | 内容 | 时间 |
|------|------|------|
| 0 | 神经网络基本单元 | 3-5 天 |
| 1 | 前向传播 | 5-7 天 |
| 2 | 反向传播 | 7-10 天 |
| 3 | RNN 与 LSTM | 7-10 天 |
| 4 | 注意力机制 | 10-14 天 |
| 5 | Transformer 深入 | 10-14 天 |
| 6 | CNN 与视觉 | 7-10 天 |
| 7 | 高级专题 | 按需 |

**总计核心内容：约 2-3 个月**（每天 2-3 小时）

---

## 检验总清单

### 基础能力（必须全部达到）
- [ ] 能手算单层网络的前向和反向传播
- [ ] 能追踪任意网络的维度变化
- [ ] 能解释梯度消失和梯度爆炸

### 核心模型（必须掌握）
- [ ] 能从零实现 LSTM（写出四个门的公式）
- [ ] 能从零实现 Attention（包括 scaled dot-product 和 multi-head）
- [ ] 能从零实现 Transformer encoder
- [ ] 能从零实现卷积层

### 理解深度（真正理解的标志）
- [ ] 能回答"为什么这样设计"的问题
- [ ] 能指出每个设计选择的 trade-off
- [ ] 能把复杂概念讲给不懂的人听

---

**"眼过千遍不如手过一遍"**

每个习题都要真正动手做，不要只是看答案！
