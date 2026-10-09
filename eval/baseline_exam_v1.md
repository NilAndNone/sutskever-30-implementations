# Sutskever 30 · 摸底考试 v1（Baseline）

- 满分 100 分，4 种题型：单选 30 分、多选 30 分、填空 20 分、简答 20 分。建议闭卷、限时 120 分钟，不查资料、不跑代码。
- 每题标了难度（L1 基础 / L2 理解 / L3 深入）和对应论文编号（P#，即仓库 notebook 编号）。30 篇论文每篇至少一题。
- **每题末尾写一个置信度 C1/C2/C3**（1=瞎蒙，2=大致有把握，3=确定）。不扣分，但"C3 且答错"会被标成误解，优先补。
- 不会就写"不会"，别蒙。蒙对的分会把 baseline 抬虚。
- 作答格式示例：`S1 B C3`、`M1 ACD C2`、`F1 …… C2`、`Q1 ……（正文） C1`。可以直接贴在线程里，也可以填好本文件后上传。
- 标"代码"的题，片段**原样摘自本仓库**（只省略了无关行）。

---

## 一、单选题（15 题 × 2 分 = 30 分）

**S1** [L1, P13] scaled dot-product attention 中除以 √d_k 的主要原因是：
A. 防止 q·k 在浮点数下溢出为 inf
B. 让 q·k 的方差不随 d_k 增长，避免 softmax 饱和导致梯度过小
C. 保证每行注意力权重之和为 1
D. 把计算复杂度从 O(n²d) 降到 O(n²)

**S2** [L2, P11] 三层 3×3 卷积，stride=1，dilation 依次为 1、2、4，堆叠后单个输出像素的感受野边长是：
A. 7　B. 13　C. 15　D. 21

**S3** [L1, P9] GPipe 用 K=4 个设备、M=8 个 micro-batch 时，pipeline bubble 占比约为：
A. 3/8　B. 1/4　C. 4/12　D. 3/11

**S4** [L2, P3, 代码] `03_lstm_understanding.ipynb` 的初始化：

```python
self.Wf = np.random.randn(hidden_size, concat_size) * 0.01
self.bf = np.zeros((hidden_size, 1))
# Wi/bi、Wc/bc、Wo/bo 同样初始化
```

训练刚开始时，信息沿 cell state 传播 20 步后大约保留原来的：
A. 0.5　B. 1×10⁻²　C. 2×10⁻³　D. 1×10⁻⁶

**S5** [L1, P4, 代码] `04_rnn_regularization.ipynb`：

```python
def dropout(x, dropout_rate=0.5, training=True):
    """
    During training: randomly zero elements with probability dropout_rate
    During testing: scale by (1 - dropout_rate)
    """
    if not training or dropout_rate == 0:
        return x
    mask = (np.random.rand(*x.shape) > dropout_rate).astype(float)
    return x * mask / (1 - dropout_rate)
```

`dropout(x, 0.5, training=False)` 实际返回：
A. x　B. 0.5·x　C. 2·x　D. 全零

**S6** [L2, P21] CTC 中目标序列为 "hello"，插入 blank 后扩展序列的长度 S 是：
A. 5　B. 10　C. 11　D. 12

**S7** [L2, P21] 要让 CTC 输出 "hello"，输入至少需要多少帧 T？
A. 6　B. 5　C. 10　D. 11

**S8** [L1, P6] Pointer Network 每一步的输出分布定义在：
A. 固定大小的输出词表上
B. 当前输入序列的各个位置上
C. 固定词表与输入位置的并集上
D. decoder 隐状态的各个维度上

**S9** [L2, P22] Kaplan et al. (2020) 中 L(N) = (N_c/N)^{α_N} 的 α_N 大约是：
A. 0.076　B. 0.34　C. 0.50　D. 0.73

**S10** [L2, P22] 关于"给定算力 C 下参数量 N 与数据量 D 如何分配"，Chinchilla (Hoffmann et al. 2022) 的结论是：
A. 主要加大模型，数据量几乎不用增长
B. 主要增加数据，模型规模保持不变
C. N 与 D 大致同比例增长，约 20 个 token/参数
D. 只要训练到收敛，分配方式无关紧要

**S11** [L2, P28] DPR 只用 in-batch negatives（不加 BM25 hard negative），batch size 为 B 时，每个问题有多少个负例？
A. 1　B. B　C. 2B−1　D. B−1

**S12** [L2, P15] 把残差改成 x_{l+1} = λ·x_l + F(x_l)，取 λ=0.5，从第 L 层到第 l 层相隔 50 层。梯度沿 shortcut 直通路径会被乘上的因子约为：
A. 0.5　B. 25　C. 8.9×10⁻¹⁶　D. 1

**S13** [L2, P25] 用 zlib 压缩后的长度来估计字符串 x 的 Kolmogorov 复杂度 K(x)，得到的是：
A. K(x) 的上界（差一个常数）
B. K(x) 的下界
C. K(x) 的精确值
D. 与 K(x) 无确定关系

**S14** [L1, P30] "Lost in the Middle" 中 U 形曲线的含义是：
A. 上下文越长，准确率越高
B. 检索文档数量与准确率呈 U 形关系
C. 模型规模与准确率呈 U 形关系
D. 含答案的文档放在上下文开头或结尾时准确率高，放在中间时最低

**S15** [L1, P2, P26] softmax + cross-entropy 对 logits z 的梯度是（p 为 softmax 输出，y 为 one-hot 标签）：
A. y − p　B. p − y　C. p(1 − p)　D. −log p

---

## 二、多选题（10 题 × 3 分 = 30 分）

全对 3 分；少选且没有错选 1 分；有错选 0 分。

**M1** [L2→L3, P13, 代码] `13_attention_is_all_you_need.ipynb`：

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + (mask * -1e9)
    attention_weights = softmax(scores, axis=-1)
    return np.dot(attention_weights, V), attention_weights

def create_causal_mask(seq_len):
    return np.triu(np.ones((seq_len, seq_len)), k=1)

class TransformerBlock:
    def forward(self, x, mask=None):
        attn_output = self.attention.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_output)
        ff_output = self.ff.forward(x)
        x = self.norm2.forward(x + ff_output)
        return x
```

以下说法正确的是：
A. 这里 mask 中 1 表示屏蔽，与 `create_causal_mask` 的约定一致
B. 这个 block 是 Pre-LN 结构
C. 如果 mask 和 scores 都是 fp16，`mask * -1e9` 会让本应可见的位置变成 NaN
D. 去掉位置编码、不加 mask 时，self-attention 对输入序列是置换等变的
E. 去掉位置编码、加上 causal mask 后，self-attention 仍然是置换等变的

**M2** [L2, P10, 代码] `10_resnet_deep_residual.ipynb`：

```python
class PlainLayer:
    def forward(self, x):
        self.z = np.dot(self.W, x) + self.b
        self.a = relu(self.z)
        return self.a

class ResidualBlock:
    def forward(self, x):
        out = self.layer1.forward(x)
        out = self.layer2.forward(out)
        self.out = out + x
        return self.out
```

以下说法正确的是：
A. 残差分支 F(x) 的每个元素都 ≥ 0
B. 这个 block 无法让输出的任何一个坐标小于输入
C. 它与 He et al. (2016) 原版残差块的结构一致
D. 去掉 layer2 在相加之前的 ReLU，是一个合理的修正
E. 因为有 skip connection，F(x) 的取值范围不影响表达能力

**M3** [L2, P4] 关于 RNN 的 dropout，正确的是：
A. inverted dropout 在推理时需要把激活乘以 (1−p)
B. Zaremba et al. (2014) 只在非循环连接上使用 dropout
C. Gal & Ghahramani 的 variational dropout 在同一序列的所有时间步共用同一个 mask
D. Zaremba et al. (2014) 在 h_{t−1}→h_t 上每个时间步独立采样 mask
E. variational dropout 可以用在循环连接上

**M4** [L1, P7] 以下哪些是 AlexNet (2012) 论文中实际使用的设计？
A. ReLU 激活
B. 全连接层的 dropout
C. Batch Normalization
D. 随机裁剪 + 水平翻转的数据增强
E. 残差连接
F. 把模型切分到两块 GPU 上训练

**M5** [L2, P17] 关于 VAE，正确的是：
A. ELBO = E_{q(z|x)}[log p(x|z)] − KL(q(z|x) ‖ p(z))
B. log p(x) − ELBO = KL(q(z|x) ‖ p(z|x))
C. q=N(μ, diag(σ²)) 与 p=N(0, I) 的 KL 为 ½ Σ (μ² + σ² + log σ² − 1)
D. 重参数化技巧的目的是让采样得到的 z 对 μ、σ 可导
E. decoder 使用很强的自回归模型（如 PixelCNN）时，z 会被更充分地利用

**M6** [L3, P20] 关于 Neural Turing Machine 的寻址，正确的是：
A. 顺序为 content → interpolation → shift → sharpen
B. content 寻址使用 key 与 memory 行之间的欧氏距离
C. sharpen（γ ≥ 1）用来抵消 shift 卷积造成的模糊
D. interpolation gate g=0 时，只使用 content 寻址的结果
E. shift 是对权重向量做循环卷积

**M7** [L3, P29, 代码] `29_rag.ipynb`：

```python
class RAGSequence:
    def forward(self, query_tokens, target_tokens, document_embeddings, documents_tokens, k=5):
        """P(y|x) = Σ_z P(z|x) * P(y|x,z)"""
        query_emb = self.retriever.encode_query(query_tokens)
        doc_indices, doc_probs = self.retriever.retrieve(query_emb, document_embeddings, k=k)
        total_prob = 0
        for doc_idx, p_z_given_x in zip(doc_indices, doc_probs):
            doc_tokens = documents_tokens[doc_idx]
            log_p_y_given_xz = self.generator.generate_prob(query_tokens, doc_tokens, target_tokens)
            p_y_given_xz = np.exp(log_p_y_given_xz)
            total_prob += p_z_given_x * p_y_given_xz
        return np.log(total_prob + 1e-8), doc_indices, doc_probs
```

以下说法正确的是：
A. 目标长 300 个 token、平均每 token log-prob 为 −3 时，返回值约为 −18.42，与输入无关
B. 目标只有 50 个 token（同样平均 −3）时，这段代码的结果是正确的
C. RAG-Sequence 允许同一条输出里不同 token 依赖不同文档
D. 改写为 logsumexp_z( log p(z|x) + log p(y|x,z) ) 可以修复数值问题
E. RAG-Token 对每个 token 分别在文档上做边缘化

**M8** [L3, P27] 同事用本仓库的 `MultiTokenRNN`（共享 RNN 主干 + 3 个线性头预测 t+1、t+2、t+3），在 20 个 token 长的合成序列上对比单 token 模型，结论是"多 token 预测让训练快了 2–3 倍，应该在所有模型上用"。以下哪些是成立的批评或事实？
A. MTP 的总 loss（各 head 相加）可以直接和单 token 模型的 loss 比较
B. 原论文里约 3 倍的加速指的是推理时用额外 head 做 self-speculative decoding
C. MTP 让每个训练 step 的计算量减少
D. 应该在相同 FLOPs 或相同参数量下比较
E. 原论文报告 MTP 的收益随模型规模增大，小模型上可能不明显甚至为负

**M9** [L2, P8, P12, P16] 以下哪些输出对输入集合中元素的排列顺序**不变**？
A. LSTM encoder 顺序读入后的最后一个隐状态
B. Relation Network：f_φ(Σ_{i,j} g_θ(o_i, o_j))
C. MPNN 中对所有节点求和的 readout
D. Pointer Network decoder 输出的下标序列
E. "Order Matters" 中 Process 模块对 memory 做 content-based attention 后得到的集合表示

**M10** [L2, P1, P5, P19, P23, P24] 以下说法正确的是：
A. Kolmogorov 复杂度 K(x) 不可计算
B. Legg & Hutter 的 universal intelligence 用 2^{−K(μ)} 对环境 μ 加权
C. 咖啡与奶混合时，粗粒化后的"表观复杂度"随时间单调上升
D. 两段式 MDL 只需要最小化 L(D|H)
E. Hinton & van Camp (1993) 的目标可以写成 KL(q(w) ‖ p(w)) + E_{q(w)}[−log p(D|w)]

---

## 三、填空题（10 题 × 2 分 = 20 分）

**F1** [L1, P3] LSTM cell state 更新：c_t = ________________。

**F2** [L1, P14] Bahdanau 加性 attention 的打分函数：e_ij = ________________（用 decoder 状态 s_{i−1} 和 encoder 注释 h_j 表示）。

**F3** [L2, P16] Relation Network：RN(O) = ________________；对物体数 n 的计算复杂度为 O(____)。

**F4** [L2, P15] Pre-activation ResNet 中 x_{l+1} = x_l + F(x_l)，则 x_L = x_l + ________________；∂E/∂x_l = ∂E/∂x_L · ( ________________ )。

**F5** [L1, P9] GPipe 的 bubble 占比公式（K 个设备、M 个 micro-batch）：________________。

**F6** [L2, P17] 一维情形 KL( N(μ, σ²) ‖ N(0, 1) ) = ________________。

**F7** [L2, P21] CTC 前向变量递推：α(t, s) = ________________；允许从 s−2 跳转的条件是 ________________。

**F8** [L2, P24] Legg & Hutter 的 universal intelligence：Υ(π) = ________________。

**F9** [L2, P12] MPNN 框架的三个函数依次是 ________、________、________。

**F10** [L2, P18] Relational RNN 的 relational memory core 中，attention 的 query 来自 ________，key/value 来自 ________。

---

## 四、简答题（4 题 × 5 分 = 20 分）

写关键步骤，不要只写结论。

**Q1** [L3, P5, P23, P25] 描述长度。
(a) 写出两段式 MDL 的目标，解释每一项的含义。
(b) 说明 Hinton & van Camp (1993) 的目标和 VAE 的 ELBO 是什么关系。
(c) 为什么 Kolmogorov 复杂度不可计算？给出论证思路。

**Q2** [L3, P17] Variational Lossy Autoencoder 关注的现象：decoder 用很强的自回归模型时，z 常常被完全忽略（posterior collapse）。为什么会这样？VLAE 怎么利用这一点来控制 z 学到什么？

**Q3** [L3, P13, P15] 第 M1 题里的 block 是 Post-LN。说明 Post-LN 和 Pre-LN 在梯度传播上的区别、为什么 Pre-LN 在深层模型上更稳定，以及它和 P15（pre-activation ResNet）是什么关系。

**Q4** [L3, P28, P30] 你要评估一个新的 128k 上下文模型是否存在 lost-in-the-middle。写一个评测方案，至少说清楚：自变量与控制变量、干扰文档怎么构造、必须有哪些基线、怎么防数据污染、怎么报告结果（包括统计）。

---

*交卷后我会逐题批改，按论文簇给出分项得分率和薄弱点清单，作为后续学习的 baseline。*
