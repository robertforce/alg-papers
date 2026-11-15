# MFU（Model FLOPs Utilization，模型浮点运算利用率）详解：计算原理与推荐模型 MFU 偏低的根因

> 类型：技术概念解析（通用原理 + 论文事实 + 说明性示例）
> 相关概念：GEMM（通用矩阵乘法）、HFU、Arithmetic Intensity、FLOPs 计数、Roofline 模型、Parameter Server、Embedding Lookup

---

## 1. MFU 是什么

**MFU（Model FLOPs Utilization）** 衡量的是：**"模型真正需要的理论计算量"对"硬件理论峰值算力"的利用效率**。

它由 PaLM 论文（Chowdhery et al., 2022, *PaLM: Scaling Language Modeling with Pathways*, Appendix B）明确提出并推广，用来横向对比不同训练系统（不同芯片、不同并行策略、不同框架）的训练效率。

与它相对的是 **HFU（Hardware FLOPs Utilization）**：

| 指标 | 分子 | 含义 |
|---|---|---|
| **HFU** | 硬件上**实际执行**的所有浮点指令数（profiler 统计到的 kernel FLOPs） | 硬件指令发射层面的利用率 |
| **MFU** | **模型理论上需要的最小** FLOPs（按公式估算，如 6N） | "有效计算"层面的利用率 |

HFU 通常明显高于 MFU，因为实际执行中还包含 recompute（重计算/梯度检查点）、稠密化的稀疏计算、LayerNorm/Softmax 等公式没计入的算子、精度转换等"额外 FLOPs"。MFU 把这些全部视为浪费，只认"公式上的必要计算"，所以更能反映**端到端系统效率**。

---

## 2. 计算原理

### 2.1 通用公式

```
MFU = 实际达到的吞吐 × 单样本理论 FLOPs ÷ (GPU 数量 × 单卡峰值 FLOPS)
```

对 LLM 训练（按 token 计）：

```
MFU = (tokens/s × FLOPs/token) / (N_gpu × 峰值FLOPS_per_GPU)
```

### 2.2 FLOPs 口径约定

必须先统一两个口径，否则数字会差一倍：

1. **1 MAC = 2 FLOPs**：一次乘加（multiply-accumulate）算 2 个浮点运算。矩阵乘法按此口径计数。
2. **训练总 FLOPs = 3 × 前向 FLOPs**：前向 1 份（2N），反向 2 份（4N，包括激活梯度与参数梯度）。

> 注意：也有论文（如 OpenAI 的 scaling law 分析）把 "C = 6ND" 里的 C 直接称为训练计算量，即每参数每 token 约 6 FLOPs，与本口径一致。若只算前向，则为 2N。

### 2.3 Transformer 每 token 的理论 FLOPs

设模型参数量为 N，层数为 L，hidden size 为 h，序列长度为 s。

**训练（前向 + 反向）：**

```
FLOPs/token ≈ 6N + 12·L·h·s
```

- `6N`：全部线性层（QKVO、FFN）的矩阵乘。前向每参数 2 FLOPs（1 MAC），反向 4 FLOPs，合计 6N。
- `12·L·h·s`：attention 的 QK^T 与 A·V 两个矩阵乘。前向每层每 token 为 `4·s·h`（两次 `s×h` 的 MAC），乘以 3 倍（含反向）得 `12·L·h·s`。

**推理（仅前向）：**

```
FLOPs/token ≈ 2N + 4·L·h·s
```

> 说明：这只是一个**下界口径**——Softmax、LayerNorm、残差、激活函数、embedding 查表等都不计入。embedding 查表（gather）本身没有浮点乘加，FLOPs 记 0，这一点是后文理解推荐模型 MFU 的关键。

### 2.4 计算示例（说明性）

以一个 7B 参数、L=32、h=4096、s=4096 的 LLaMA 级模型为例：

- `6N = 6 × 7×10⁹ = 4.2×10¹⁰` FLOPs/token
- `12·L·h·s = 12 × 32 × 4096 × 4096 ≈ 6.4×10⁹` FLOPs/token
- 合计 ≈ `4.85×10¹⁰` FLOPs/token（attention 项约占 13%，s 较小时 `6N` 完全主导）

在 8 张 A100（BF16 峰值 312 TFLOPS/卡，合计 2496 TFLOPS）上实测吞吐 100,000 tokens/s：

```
MFU = (1.0×10⁵ × 4.85×10¹⁰) / (2.496×10¹⁵) ≈ 1.94%
```

咦，这个数明显偏低——因为示例配置下每 token 计算量对 8 卡来说太小。换成 8 张卡上实际能跑到 ~50 万 tokens/s 的大 batch 场景：

```
MFU = (5.0×10⁵ × 4.85×10¹⁰) / (2.496×10¹⁵) ≈ 9.7%
```

仍不算高，说明 7B 模型 + A100 的组合里，模型太小、卡太多时 MFU 天然上不去——**MFU 对"计算密度"极其敏感**，这本身就是理解推荐模型问题的伏笔。

一个更贴近论文事实的数字：**PaLM 540B 在 6144 块 TPU v4 上训练，MFU = 46.2%，HFU = 65.8%**。这是 MFU 的经典参考值；业界优化良好的 LLM 训练 MFU 一般在 **35%–55%** 区间（如 Megatron-LM 报告约 50% 上下，Llama 3 405B 报告 H100 上约 38%–43%）。

### 2.5 MFU 低的三类根因（分析框架）

```
MFU = 实际吞吐 × 单样本理论FLOPs / 峰值算力
```

把分子分母拆开，任何 MFU 损失都来自三类问题：

1. **算子层：kernel 效率低**（算子形状小 → Tensor Core 填不满；访存受限 → 算得快不如搬得快，GEMM 层面的机理详见第 3 节）；
2. **模型层：计算密度低**（模型本身每样本 FLOPs 太小，或大量操作不产生 FLOPs）；
3. **系统层：流水线气泡**（等数据、等通信、CPU-GPU 传输、host 侧开销，GPU 在"空转"）。

推荐模型在这三层全部中招，LLM 则在系统层做了大量针对性优化。

---

## 3. GEMM（通用矩阵乘法）：理解算子效率的基石

GEMM（General Matrix Multiplication，通用矩阵乘法）是 BLAS（Basic Linear Algebra Subprograms，基础线性代数子程序）标准中的 **Level 3** 例程，也是深度学习负载的核心：Transformer 的全部线性层、attention 的 QKᵀ 与 A·V、全连接层、1×1 卷积（经 im2col 或直接映射），最终都由一次或多次 GEMM 完成。前文公式里的 `6N` 与 `12·L·h·s`，本质上就是若干次 GEMM 的 FLOPs 求和。

### 3.1 定义与 BLAS 三级例程

**GEMM 的计算定义**：

```
C = α·A×B + β·C
A ∈ R^(M×K)，B ∈ R^(K×N)，C ∈ R^(M×N)
FLOPs = 2·M·N·K    （M·N·K 次 MAC，每次乘加计 2 FLOPs）
```

习惯上用三元组 `(M, N, K)` 描述一个 GEMM 的形状。深度学习中的对应：M = batch × token 数（输出行数），K = 输入特征维度，N = 输出特征维度。

**为什么 GPU 的算力押注在 GEMM 上？看 BLAS 三级的算术强度对比**：

| BLAS 级别 | 典型例程 | 计算量 | 访存量 | 算术强度 |
|---|---|---|---|---|
| Level 1（向量-向量） | AXPY: `y ← αx + y` | O(n) | O(n) | O(1) |
| Level 2（矩阵-向量） | GEMV: `y ← Ax` | O(m·n) | O(m+n) | O(1) |
| Level 3（矩阵-矩阵） | GEMM: `C ← AB` | O(m·n·k) | O(mn+nk+mk) | **O(维度)** |

只有 Level 3 的计算量随维度**立方**增长、访存量只随**平方**增长，数据可以被充分复用，算术强度随维度线性上升——这是 GPU 把峰值 FLOPS 押注在 GEMM 上、深度学习框架把一切计算尽量表达成 GEMM 的根本原因，也是"算子能不能写成大 GEMM"成为性能分水岭的原因。

### 3.2 GEMM 的算术强度：由最小维度决定

FLOPs 与 HBM 访存量（按 BF16、每元素 2 字节；读 A、B 各一次，C 读一次写一次）：

```
FLOPs = 2·M·N·K
访存 ≈ 2·(MK + NK + 2MN) 字节
AI   = MNK / (MK + NK + 2MN)   （FLOPs/byte）
```

**说明性示例**（设 M=N=K=n，则 AI ≈ n/4）：

- `n = 4096`（LLM 训练/prefill 的 FFN 层，M = batch×seq 上万）：AI ≈ **1024 FLOPs/byte**，远超 A100 BF16 平衡点（312×10¹² / 2039×10⁹ ≈ 153 FLOPs/byte）→ **compute-bound，能吃满算力**。
- `M = 8, N = K = 4096`（LLM decode 单步投影的典型形状）：FLOPs ≈ 2.7×10⁸，访存 ≈ 3.4×10⁷ 字节，AI ≈ **8 FLOPs/byte** → **memory-bound**，利用率被带宽卡死。

由此得到一条实用规律：**GEMM 的算术强度近似由最小的那个维度决定**（AI ≈ min(M, N, K) 的量级——当 M 很小时，分母被 B 矩阵的 NK 项主导，AI ≈ M）。所以"batch 大不大、序列长不长"（决定 M）直接决定算子落在 Roofline 哪一段：

- LLM 训练 / prefill：M = batch × seq_len 轻松上万 → 大 GEMM → compute-bound；
- LLM decode：M = batch（几~几百）→ 退化为近似 GEMV → memory-bound，这正是 **continuous batching 的动机：把有效 M 堆上去**；
- 推荐模型 dense 层：M = batch（几百~几千），但 K、N 只有几十~一千出头，embedding dim 常见 8~64，min 维度极小 → AI 个位数到几十 → memory-bound。

### 3.3 GPU 如何执行 GEMM：分块、Tensor Core 与小 GEMM 的三类损失

主流实现（NVIDIA 上的 cuBLAS、CUTLASS，以及各芯片等价库）将 GEMM **分块（tiling）**执行：

1. 把 C 按 tile（如 128×128）分块，A、B 对应的 K 向切片（如 128×32、32×128）分批搬入 **shared memory**；
2. 线程块内的 warp 调用 **Tensor Core** 的 MMA 指令（如 16×8×16 的矩阵片乘加）做累加；
3. 累加完成的 C tile 写回 HBM。

一个 GEMM 想接近峰值算力，需要同时满足：

- **tile 数量足够多**：把所有 SM（A100 有 108 个）填满并排满多个 wave；
- **每个 tile 的 M、N、K 都足够大**（经验上 ≥ 数百）：Tensor Core 流水线才能满负荷。

**小 GEMM 的三类典型损失**：

1. **tile 填不满**：M=8 时，128×128 的 tile 里约 94% 是无效填充（或根本走不到 Tensor Core 路径），有效算力比例极低；
2. **launch/调度开销占比失控**：一次 kernel launch 约几微秒，小 kernel 本身只执行几十微秒，launch、显存分配、shape 推断等 host 开销轻松吃掉 MFU；
3. **尾波效应（wave quantization）**：tile 总数除不尽 SM 数时，最后一个 wave 里大部分 SM 空转——形状不规则的 GEMM 几乎必然踩中。

反过来看，FlashAttention 等 fused kernel 的本质，是把"多个小算子"重写为"少数几个访存可控的大 GEMM 复合体"，这正是它快的原因，也是 LLM 生态持续优化的主战场；而推荐模型的大量细碎算子，尚未等到同等级别的融合基建。

### 3.4 GEMM 视角回看 MFU 公式

MFU 里的 `6N` 可以由 GEMM 公式直接推出：线性层权重 W ∈ R^(K×N)，对 T 个 token 做一次投影的 FLOPs = 2·T·K·N，即**每参数每 token 前向 2 FLOPs**，含反向 ×3 → 6N。attention 项 `12·L·h·s` 同理，是 QKᵀ（`s×d_head @ d_head×s`）与 A·V（`s×s @ s×d_head`）两组 batched GEMM 的求和。

更重要的是，下一节的结论在这个视角下变得自然：

- **LLM 的快**：由少数形状巨大、静态可预测的 GEMM 构成，每个 GEMM 都处在 Roofline 平台段；
- **推荐模型的慢**：由大量形状小、动态变化的 GEMM + 零 FLOPs 的 gather（embedding lookup）构成，每个算子都落在 Roofline 斜坡段；
- **生成式推荐（HSTU/OneRec 类）的效率红利**：本质上是把负载从"小 GEMM + 查表"替换为"大 GEMM"，让同样的硬件第一次有机会跑进 compute-bound 区间。

---

## 4. 为什么推荐算法模型的 MFU 显著低于 LLM

工业界实践与多篇系统论文（Meta DLRM、百度 PaddleBox、阿里 XDL/DeepRec、字节与快手的大规模稀疏训练系统报告等）的共同结论是：**大规模稀疏推荐模型（CTR/CVR、排序、召回）训练的 GPU 有效利用率通常只有个位数百分比，优化好的密集计算部分也难超 20%–30%；而 LLM 训练普遍 40%+**。推理侧差距更大：推荐单请求推理 MFU 常低于 1%–5%，LLM 推理在 continuous batching 下可达 20%–50%（prefill 阶段更高）。

根本原因可以概括为一句话：**LLM 是"计算密集型"（compute-bound）的稠密大矩阵计算，推荐模型是"访存密集型"（memory-bound）的稀疏查表 + 小矩阵计算**。下面逐条展开。

### 4.1 Embedding 参数占绝对多数，但查表不产生任何 FLOPs

这是最本质的一条，直接作用在 MFU 的分子上。

- 典型工业级 CTR 模型：**总参数 10¹⁰ ~ 10¹² 级别，其中 embedding table 占 99% 以上**，dense 网络（MLP、attention）通常只有 10⁶ ~ 10⁸ 参数。
- embedding lookup 是 **gather 操作**：根据 ID 从大表里拷出若干行向量，**没有一次浮点乘加，FLOPs = 0**，但消耗大量内存带宽和随机访问延迟。
- 对参数量在 10¹¹ 的推荐模型按 `6N` 套公式会得到荒谬的"理论 FLOPs"（每 token 6×10¹¹，比 GPT-4 级别还高），而实际每样本 dense 计算 FLOPs 往往只有 **10⁵ ~ 10⁷** 量级——**分子与分母完全失配，MFU 这个指标对推荐模型的定义本身就不成立**。

> 这就是为什么推荐系统领域几乎不用 MFU 做核心指标，而是用 **samples/s（训练吞吐）、QPS 与 P99 延迟（推理）、GPU SM 占用率、端到端耗时**来评估。MFU 偏低在推荐场景很大程度上是**指标不适配**的表征，背后是真实的算力浪费。

### 4.2 算子形状小且不规则，喂不饱 Tensor Core

GPU 的峰值 FLOPS（如 A100 BF16 312 TFLOPS）是为**大 GEMM** 设计的，只有当 M、N、K 三个维度都足够大（经验上 ≥ 数百到上千）时，Tensor Core 的流水线才能被打满。

对比两类负载：

| 维度 | LLM | 推荐模型 |
|---|---|---|
| 主要 GEMM 形状 | `(batch × seq_len) × h × h²`，M 动辄数万（如 batch=32, s=4096 → M=131072） | 单样本/小 batch 上的 `batch × d_in × d_out`，M 常常 < 512 |
| hidden dim | 4096 ~ 16384 | 特征 embedding dim 常见 8 ~ 64，MLP 宽度 64 ~ 1024 |
| 算子数量 | 少而大（每层 4~6 个大 GEMM） | 多而小：上百个特征域各自 lookup + 变换 + 池化，再接小 MLP |
| kernel 形状 | 静态、可预测 | **动态**：batch 内序列长度、特征数、候选 item 数每步都变 |

小 GEMM 的问题：Tensor Core 利用率低、kernel launch 开销占比高（一次 launch ~几微秒，而小 kernel 执行可能只有几十微秒）、无法有效融合。LLM 的形状静态性还使得 CUDA Graph、torch.compile、XLA 等图编译技术能稳定生效；推荐模型的**动态 shape 是图编译优化的天敌**，大量时间花在 launch 和调度上。

### 4.3 计算强度（Arithmetic Intensity）低，天生平是 memory-bound

Roofline 模型：`可达FLOPS = min(峰值FLOPS, 带宽 × 算术强度)`，其中算术强度 AI = FLOPs / 访存字节数。

- LLM 的 encoder/FFN GEMM：AI 在数百 FLOPs/byte 量级以上（h=4096 的方阵 GEMM 理论 AI ≈ h/4 ≈ 10³，见第 3.2 节），远高于 A100 的平衡点（FLOPS/带宽 ≈ 312×10¹² / 2039×10⁹ ≈ 153），属于 compute-bound，**能吃满算力**。
- 推荐 dense 部分：`d_in × d_out` 的小矩阵乘，AI 常在 **个位数到几十**，落在 memory-bound 区间，算力利用率被带宽卡死。
- embedding lookup：AI ≈ **接近 0**（拷贝 4~32 字节 × dim，0 FLOPs），纯带宽/延迟型负载。

也就是说，**推荐模型的每个组成部分都处在 Roofline 的斜坡段**，而 LLM 的主体算子处在平台段。

### 4.4 系统层：数据管线、稀疏通信与 CPU-GPU 协同带来大量气泡

MFU 的分母是"峰值算力 × 时间"——GPU 只要有一纳秒没在算有效计算，MFU 就在损失。推荐训练的系统开销远比 LLM 训练复杂：

1. **数据加载与预处理瓶颈**：训练样本包含数百个稀疏特征（多值、变长），样本解析、特征拼接、序列 padding、shuffle 等在 CPU 上进行，单条样本处理耗时常超过 GPU 计算时间，GPU 频繁"等米下锅"。
2. **稀疏参数的存储与通信**：embedding 表大到单卡 HBM（40~80GB）装不下，必须分片到多机。LLM 训练的通信主要是规整的 **All-Reduce**（梯度同步，可与反向计算 overlap）；推荐训练则是**不规整的 All-to-All / 参数服务器 Pull-Push**，且每次拉的行随 batch 变化，通信粒度碎、难以与计算重叠，还常需访问远端 KV 存储/SSD（缓存 miss 时网络 RTT 直接变成 GPU 空闲）。
3. **CPU-GPU 传输**：特征 ID 与非数值特征在 CPU 侧处理后经 PCIe 拷贝到 GPU，PCIe 带宽（~32~64 GB/s）远低于 HBM，容易成为瓶颈。
4. **部署异构**：很多工业方案干脆把 embedding 放 CPU 或专用存储（PaddleBox、XDL、DeepRec 的 PS 架构），GPU 只跑 dense 部分——此时看 GPU 的"MFU"更低，但这是**架构选择**（用便宜的 CPU 内存放超大表）而非纯粹的低效。

LLM 训练侧这些系统问题同样存在，但因为**每步计算量巨大**（一次 step 的 dense FLOPs 是推荐模型的成千上万倍），通信和数据开销可以被计算完全掩盖（overlap），气泡占比小。

### 4.5 模型结构层面：控制流多、算子碎、负载不均衡

- 多任务学习（shared-bottom / MMoE / PLE）：多个 tower、多个 loss、gate 网络带来大量小算子和分支控制流。
- 目标 attention（如 DIN）对每个候选 item 单独计算，是典型的 **batch 内动态循环**，算子粒度极细。
- 负采样、样本加权、多目标 loss mask 等逻辑引入不规则张量操作。
- 相比之下 LLM 是**同构层的规则堆叠**（几十上百层完全相同的 transformer block），天然适合 kernel 融合（FlashAttention 这类 fused kernel）与流水线并行。

### 4.6 推理侧为什么差距更大

**LLM 推理：**
- 单请求 FLOPs 巨大（每个 token 前向 `≈2N`，7B 模型每 token 1.4×10¹⁰ FLOPs），即使 decode 阶段 memory-bound，经 **continuous batching** 聚合后 GPU 仍能维持较高带宽与算力利用率；prefill 阶段是大 GEMM，compute-bound，MFU 可达 40%+。
- 优化栈成熟：PagedAttention、量化（INT8/FP8/AWQ）、CUDA Graph、KV cache 复用。

**推荐推理：**
- **单请求计算量极小**：一次请求只对一批候选（几十~几千个）各算一个分数，dense 前向 FLOPs 约 10⁵ ~ 10⁷ 量级，**GPU 执行时间常常只有几十~几百微秒**，而延迟预算是 10~50ms——**绝大多数时间花在 embedding 查表、特征拼接、网络传输和排队上**，算力利用率必然极低。
- **延迟约束压制 batching**：在线服务要求 P99 长尾可控，不能像 LLM 那样无限积攒大 batch；而推荐请求间特征与候选数高度异构，batch 内动态 shape 使 padding 浪费和 kernel 低效叠加。
- **特征工程在 CPU**：请求级特征抽取、ID 哈希、交叉特征计算都在 host 侧，GPU 只是流水线末端的一小段。
- **集群经济学反差**：LLM 推理一台 8 卡机服务有限并发，GPU 理所当然被算力打满；推荐推理 QPS 极高但单请求太轻，部署上甚至大量使用 CPU 推理（算力需求小、embedding 表大、扩容便宜），GPU 实例也往往被 QPS 切得很碎——**MFU 低是负载特性决定的，不是"没优化好"**。

### 4.7 一张总表

| 根因 | 作用层面 | LLM | 推荐模型 |
|---|---|---|---|
| Embedding 参数多但零 FLOPs | 指标/模型 | 参数基本都参与计算 | 99%+ 参数零 FLOPs |
| 算子形状 | 硬件 | 大 GEMM，Tensor Core 打满 | 小而碎的 GEMM/动态 shape |
| 算术强度 | 硬件 | 数百~10³ FLOPs/B，compute-bound | ≈0~几十 FLOPs/B，memory-bound |
| 图编译/融合 | 软件 | 静态图，CUDA Graph/XLA 生效好 | 动态 shape，难编译融合 |
| 通信 | 系统 | 规整 All-Reduce，可与计算 overlap | All-to-All / PS Pull-Push，碎片化 |
| 数据管线 | 系统 | 文本 token 化后很轻 | 数百稀疏特征解析拼接，CPU 瓶颈 |
| 推理单请求负载 | 推理 | 每 token 10¹⁰ FLOPs 量级 | 每请求 10⁵~10⁷ FLOPs，GPU 亚毫秒级 |
| 延迟与 batching | 推理 | continuous batching 成熟 | P99 约束 + 异构请求，batch 难做大 |

---

## 5. 正确看待推荐模型的"低 MFU"

1. **先问指标适不适配**：对大规模稀疏模型，MFU 的分子（理论 FLOPs）定义不清（embedding 零 FLOPs、动态 attention 随输入变化），业界改用端到端吞吐、成本/QPS、GPU 有效计算时间占比等指标。
2. **低 MFU ≠ 低效**：把 embedding 放 CPU/远端存储是"用廉价内存换 GPU HBM"的理性架构；此时 GPU 只承担 dense 计算，表观 MFU 低但**单位成本样本数（cost per sample）**可能是最优的。
3. **真正值得优化的部分**：dense 侧的小算子融合（如把特征交互 + MLP 编译成大 kernel）、embedding 换成 GPU 端 + HBM/多卡 NVLink 分片（GPU 参数服务器路线）、动态 shape 分桶（bucketing）+ CUDA Graph、数据管线零拷贝流水。这些优化的目标不是"把 MFU 刷高"，而是降低训练成本和推理 P99。
4. **趋势交集**：生成式推荐（如 HSTU、OneRec 这类将推荐建模为序列生成的新范式）正在把推荐模型**推向 LLM 的负载形态**——大 dense transformer、规则算子、长序列 GEMM——这正是这类工作声称"训练效率大幅提升"的系统性原因之一；而传统 ID-based 稀疏模型的低 MFU 是其**稀疏查表本质**决定的，不会因为工程努力而消失。

---

## 6. 参考

- PaLM 论文（MFU/HFU 定义与 46.2% 基准）：Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways*, 2022, Appendix B。
- FLOPs 近似公式（6N 与 attention 项）：Kaplan et al., *Scaling Laws for Neural Language Models*, 2020（C=6ND）；PaLM Appendix B（12Lhs 项）。
- 推荐系统稀疏训练架构与瓶颈：Meta *DLRM: A Deep Learning Recommendation Model*（2019）；百度 *PaddleBox*（KDD 2022）；阿里 *XDL*（KDD 2019）与 *DeepRec*；相关工业界分享（字节、快手大规模稀疏训练系统）。
- LLM 推理效率：vLLM（*Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023）、TensorRT-LLM 技术报告。
- Roofline 模型：Williams et al., *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, CACM 2009。
- GEMM 与 BLAS：BLAS 标准（Lawson et al., 1979）；Goto & van de Geijn, *Anatomy of High-Performance Matrix Multiplication*, ACM TOPC 2008（分块与寄存器/缓存层级优化）；NVIDIA cuBLAS / CUTLASS 文档（tile 形状、Tensor Core MMA 与 kernel 调度）。

> 说明：文中标注"说明性"的数字（如 7B 模型示例、GEMM 算术强度示例、推荐模型 FLOPs 量级）为便于理解的量级估算，具体数值随模型结构、序列长度、batch 策略不同而变化；LLM/推荐系统的 MFU 区间为论文与公开技术报告中的典型参考值，非精确界限。
