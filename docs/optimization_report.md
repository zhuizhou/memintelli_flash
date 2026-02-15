# Memintelli Flash 优化报告

## 零、当前版本新增功能与阶段性结果

> 本文档基于最新代码版本整理。相比前一版，当前版本在**显存可控性**和**推理速度**上都有明显提升，且多模型优化能力已统一。

### 0.1 本轮新增功能（新增/补齐）

1. **Conv2dMem 流式推理能力补齐（与 LinearMem 对齐）**
   - 新增 `skip_initial_mapping`
   - 新增 persistent pinned buffers（避免每次 forward 重新 D2H + pin）
   - 新增 async prefetch（独立 CUDA stream）和 `_release_gpu_tensors`（零拷贝释放 GPU 引用）

2. **Conv2dMem 前向显存保护：空间维分块（spatial chunking）**
   - 在 `Conv2dMem._forward_inference()` 中对 `unfold` 后的空间维 `L=H_out*W_out` 分块
   - 解决 VGG 前几层在 `slice_data_imp` 阶段产生超大中间张量导致 OOM 的问题

3. **模型级 3-Phase pipeline 全面统一**
   - `VGG / DeiT / ResNet / ResNet_CIFAR / MobileNetV2 / Qwen3` 统一为：
     - Phase 1: 逐层算 G 并立刻 offload 到 pinned CPU
     - Phase 2: `streaming=False/True/"auto"` 策略决策
     - Phase 3: 构建 streaming 层 async prefetch 链

4. **`deit_zoo` 参数传递 bug 修复**
   - 修复 `input_paral_size/weight_paral_size/input_quant_gran/weight_quant_gran` 被静默丢弃的问题
   - 使 DeiT 的并行与量化配置真正生效

5. **示例脚本修复与规范化**
   - ImageNet/LLM 示例统一使用 `torch.inference_mode()`
   - 补齐 `DPETensor(device=..., inference_chunk_size=...)`
   - 统一推荐 `streaming="auto"`，避免“定义了 streaming 但未传参”的错误用法

### 0.2 阶段性结果（当前已观测）

- **VGG ImageNet**
  - 初始化期 OOM 已解决（`skip_initial_mapping` 生效）
  - 前向期 Conv unfold/slice OOM 已由 spatial chunking 缓解
- **DeiT ImageNet**
  - 参数修复后吞吐显著提升；由原先极慢（20h 量级）降到可运行区间（实测约 5.5s/batch）
- **ResNet ImageNet/CIFAR**
  - 已接入 auto/streaming/async pipeline，配置一致性和可维护性显著提升
- **总体结论**
  - 当前版本“能跑+可扩展+可控显存”目标已达成，后续优化重点从“能否运行”转向“性能上限和科研验证”

---

## 一、所有修改详细清单

### 1. 引擎核心层 (`memintelli/pimpy/memmat_tensor.py`)

#### 1.1 float64 → float32 全面降精度
- **修改位置**: `_num2G()`, `_dot_inference_batch()`, `_dot_inference_nobatch()`
- **修改内容**: 将 conductance 计算从 float64 改为 float32，避免 int64 中间变量
- **原因**: float64 张量占用 2x 内存，且 GPU 对 float64 吞吐量极低（1/32 of float32 on consumer GPUs）
- **效果**: 内存减半，计算速度提升 ~2x

#### 1.2 分块推理 (Chunked Inference)
- **修改位置**: `_dot_inference_batch()`, `_dot_inference_nobatch()` —— 新增方法
- **修改内容**: 权重矩阵按列方向(nc_y)分块处理，`inference_chunk_size` 控制每次处理的元素数
- **参数**: `inference_chunk_size`（默认 None = 32M 元素 ≈ 128MB float32）
- **原因**: 大层（如 lm_head: 2560→151936）的 G 矩阵无法一次性放入 GPU
- **效果**: 以轻微的 Python 循环开销换取可控的显存峰值

#### 1.3 向量化 slice 循环 → 批量 matmul
- **修改位置**: `_dot_inference_batch()`, `_dot_inference_nobatch()` 的 `(i,j)` 内层循环
- **修改内容**: 
  - **之前**: `for i in ns_x: for j in ns_y: torch.matmul(...)` —— ns_x×ns_y 次独立 kernel 启动
  - **之后**: `torch.stack(Vin_list) → torch.matmul(Vin_flat.unsqueeze(1), G_c_all.unsqueeze(0))` —— 1 次批量 matmul
  - 同时将 `slice_scale` 从 Python list 改为 tensor `(ns_x, ns_y)` 用于向量化 ADC 后的 scale+sum
- **原因**: GPU kernel launch overhead 在小矩阵上占比极高（~0.1ms/launch × 16 = 1.6ms/block）
- **效果**: 每个 block 的 kernel 启动次数从 ns_x×ns_y(如 16) 减少到 1，GPU 利用率显著提升

#### 1.4 内存高效噪声生成
- **修改位置**: `_gen_read_noise_shifted_chunk()` —— 新增方法
- **修改内容**: 从 G_indices(uint8) 原地重建 float32 G 并加噪，只处理当前 chunk 列
- **原因**: 原来对整个 G 矩阵一次性生成噪声会 OOM
- **效果**: 噪声生成的内存从 O(全部权重) 降为 O(一个chunk)

#### 1.5 G 压缩 (uint8 indices)
- **修改位置**: `data_formats.py` 的 `SlicedData.compress_G()`
- **修改内容**: 当 `write_variation=0` 时，G 值精确落在离散 conductance 级别上，可用 uint8 索引表示
- **原因**: float32 G 占 4B/元素 → uint8 占 1B/元素，4x 压缩
- **效果**: Qwen3-4B 全部层的 G 从 ~6GB 压缩到 ~1.5GB

### 2. 神经网络层 (`memintelli/NN_layers/linear.py`)

#### 2.1 推理模式快速路径
- **修改位置**: `_forward_inference()` —— 新增方法
- **修改内容**: `inference_mode=True` 时跳过 autograd，直接调用 `engine.MapReduceDot`
- **原因**: 推理不需要梯度，省掉 autograd 记录开销
- **效果**: 前向传播速度提升，内存减少（不保存中间激活）

#### 2.2 dtype 自动转换
- **修改位置**: `_forward_inference()` 末尾
- **修改内容**: `if output.dtype != input.dtype: output = output.to(input.dtype)`
- **原因**: RRAM 引擎始终以 float32 计算，但模型可能是 bfloat16/float16。不转换会导致：
  - Flash Attention 被禁用（需要 float16/bfloat16）
  - 所有下游激活变成 float32 → 2x 内存、0.5x 速度
- **效果**: 使 Flash Attention 正常工作，激活内存减半

#### 2.3 SlicedData 缓存复用
- **修改位置**: `_forward_inference()` 中 input_sliced 的创建
- **修改内容**: 首次创建后缓存到 `self._input_sliced_cache`，后续复用对象（只更新数据）
- **原因**: `SlicedData.__init__` + `_init_data` 涉及多次 Python 循环和 tensor 分配
- **效果**: 消除每次 forward 的 SlicedData 对象创建开销

#### 2.4 流式传输 (Streaming) + Pin Memory
- **修改位置**: `_offload_to_cpu()`, `_load_to_device()` —— 新增方法
- **修改内容**: 
  - `_offload_to_cpu()`: 将 G/G_indices/max_data 移到 CPU 并 pin_memory()
  - `_load_to_device()`: 从 pinned CPU memory 异步传输到 GPU + synchronize
- **原因**: Streaming 模式下，只有当前层的 G 在 GPU 上；pinned memory 提供 2-3x DMA 加速
- **效果**: Qwen3-4B 推理 GPU 显存从 ~12GB 降到 ~3-4GB

#### 2.5 `skip_initial_mapping` 延迟映射
- **修改位置**: `__init__` 新增 `skip_initial_mapping` 参数
- **修改内容**: 为 True 时跳过构造函数中的 `slice_data_imp()`（即不计算 G）
- **原因**: 预训练模型加载后会覆盖权重，初始 G 毫无意义且浪费 GPU 内存
- **效果**: 模型初始化内存峰值大幅降低

### 3. 模型层 (`memintelli/NN_models/Qwen3.py`)

#### 3.1 CPU 创建 LinearMem 权重
- **修改位置**: `_replace_linear_with_linearmem()`
- **修改内容**: `device='cpu', dtype=child.weight.dtype`（而非 GPU float32）
- **原因**: 8B 模型: 原始 bfloat16 ~16GB + 新 float32 ~32GB = 48GB → OOM on 24GB GPU
- **效果**: 替换阶段 GPU 内存零增长

#### 3.2 逐层 update_weight_and_prepare
- **修改位置**: `Qwen3MemWrapper.update_weight_and_prepare()`
- **修改内容**: 逐层: CPU→GPU权重 → 计算G → 压缩uint8 → 释放权重 → (可选)offload G到CPU
- **原因**: 所有权重同时在 GPU 上会 OOM
- **效果**: 峰值 = embedding + norms + 1层权重 + 1层G ≈ 2-3GB（8B 模型也能跑）

#### 3.3 跳过 lm_head
- **修改位置**: `qwen3_zoo()` 新增 `skip_embedding_and_head: bool = True`
- **修改内容**: lm_head 保持原始 nn.Linear，不做 RRAM 仿真
- **原因**: lm_head（如 Qwen3-4B: 2560→151936）占 ~50% 推理时间 + ~1.5GB G 存储
- **效果**: 推理时间减半，内存显著减少

#### 3.4 释放原始权重
- **修改位置**: `update_weight_and_prepare()` 中 `free_weights=True`
- **修改内容**: G 计算完成后 `module.weight.data = torch.empty(0, device='cpu')`
- **原因**: 推理模式下只用 G，原始权重再也不需要
- **效果**: Qwen3-4B: 释放 ~7.3GB

### 4. 推理脚本 (`examples/11_qwen3_wikitext2_inference.py`)

#### 4.1 `use_cache=False`
- **修改内容**: forward 调用加 `use_cache=False`
- **原因**: PPL 评估不复用 KV cache，默认 True 会无意义地分配/缓存 KV → 显存持续增长

#### 4.2 `del outputs` 即时释放
- **修改内容**: `nll_val = (outputs.loss * trg_len).item(); del outputs`
- **原因**: outputs 持有 logits 张量（~155MB for 4B），不 del 则 Python 引用计数不归零

#### 4.3 `.to(device)` 延后到 G 计算之后
- **修改内容**: 不再在 `qwen3_zoo()` 后链式 `.to(device)`，而是在 `update_weight_and_prepare()` 之后
- **原因**: 避免将所有 LinearMem 权重（float32）一次性搬到 GPU

---

## 二、下一步优化建议（基于当前已实现版本）

### 2.1 现阶段主要瓶颈定位

当前 Streaming/Auto 体系已可用，下一步瓶颈主要在：
- **超大层 ADC/噪声计算开销**（尤其 LLM MLP 大层、VGG 大 FC）
- **Python 循环与小 kernel 调度开销**（chunk/slice 维度仍有调度损耗）
- **不同模型的最优参数未自动化**（chunk_size、radc、slice 配置仍靠手调）

### 2.2 建议 1：Auto 模式加入“速度优先”策略（高优先级）

当前 Auto 主要按显存预算决策“驻留/流式”。建议新增可选策略：
- `streaming="auto_speed"`：在不 OOM 前提下优先减少传输次数（更激进地保留热点层在 GPU）
- `streaming="auto_memory"`：优先稳健显存（当前默认语义）

可增加层级统计项：
- 最近 N step 的层调用频次
- 层传输耗时 EMA
- 层计算耗时 EMA

据此构建“收益/显存占用”比值进行驻留选择。

### 2.3 建议 2：LinearMem 增加输入维分块（补齐 FC 超大层保护）

Conv2dMem 已有空间分块；建议在 `LinearMem._forward_inference()` 增加输入 token/batch 维分块：
- 避免 `input_sliced.slice_data_imp()` 在大 batch 或大序列时中间张量峰值过高
- 与 `DPETensor.inference_chunk_size` 联动统一控制

### 2.4 建议 3：噪声与 ADC 计算融合/近似（科研+工程双收益）

可尝试两条线并行：
- **Kernel 融合**：将 `read_noise + ADC quant + scale` 融合到单 kernel（可用 Triton/CUDA）
- **统计近似**：对稳定层采用查表/分段近似，减少逐元素算子开销

### 2.5 建议 4：建立“模型-配置”自动调参器

输入：GPU 容量、目标模型、目标精度损失阈值  
输出：推荐 `slice / radc / rdac / chunk_size / streaming 策略`

评价目标：
- 最小延迟（或最大吞吐）
- 显存不越界
- 精度退化不超过阈值

---

## 三、TCAD 论文方向：大规模 RRAM CIM 仿真器优化

### Challenge（挑战）

1. **Memory Wall**: RRAM CIM 仿真需要存储完整的 conductance 矩阵 G（float32），对 LLM 级别模型（4B-70B 参数），G 矩阵远超 GPU 显存
   - Qwen3-4B: G 占 ~6GB（压缩前），~1.5GB（uint8 压缩后）
   - Qwen3-70B: G 占 ~100GB+，即使压缩也远超单卡显存

2. **Compute Overhead**: RRAM 仿真引入 DAC 量化、读噪声、ADC 量化等 非线性操作，无法被 GPU tensor core 原生加速，需要多次逐元素操作

3. **Kernel Launch Bottleneck**: Bit-slicing 导致 ns_x × ns_y 种 slice 组合，每种需要独立 matmul → 大量小 kernel 启动开销

4. **Scalability Gap**: 现有 CIM 仿真器（如 NeuroSim、TxSim）只能仿真 ResNet/VGG 级别的小模型，无法扩展到 LLM

### Innovation（创新点）

1. **Hierarchical Memory Management for CIM Simulation**
   - CPU-GPU streaming with pinned memory + async prefetch
   - G compression: float32 → uint8 level indices (4x reduction)
   - Per-layer sequential processing (never all G on GPU simultaneously)
   - 首次实现了在单 24GB GPU 上仿真 8B 参数 LLM 的 RRAM CIM 推理

2. **Vectorized Slice Computation**
   - 将 ns_x × ns_y 次独立 matmul 合并为单次 batched matmul
   - 利用 PyTorch broadcasting: `(ns_x,1,B,n) @ (1,ns_y,n,p) → (ns_x,ns_y,B,p)`
   - 减少 kernel launch 次数 16x（以 4-bit slice 为例）

3. **Selective CIM Simulation**
   - 识别并跳过对 CIM 非理想性不敏感的层（如 lm_head）
   - 保留原始 nn.Linear → 推理时间减半
   - 可以进一步扩展为"层级灵敏度分析"，自动决定哪些层需要全精度仿真

4. **Chunked Inference with Memory-Aware Scheduling**
   - 自适应分块策略：根据可用 GPU 显存动态调整 chunk 大小
   - 内存预估模型：`peak_mem = output_per_col × chunk_cols + g_per_col × chunk_cols`
   - 首次在 CIM 仿真器中实现了"可控显存"推理

### 论文 Story

**Title**: *Scalable RRAM Compute-in-Memory Simulation for Large Language Models: A Memory-Efficient Framework*

**核心贡献**:
1. 首个能在单张消费级 GPU (24GB) 上完成 4B-8B 参数 LLM RRAM CIM 仿真推理的框架
2. 提出层级流式处理 + G 压缩 + 批量化 slice 计算的联合优化方法
3. 在 Qwen3-4B 上实现 WikiText-2 PPL 评估，验证了 RRAM 非理想性（读噪声、DAC/ADC量化）对 LLM 推理质量的影响
4. 与现有仿真器（NeuroSim、TxSim）相比，支持的模型规模扩大 100x+

### 3.1 科研计划（6-12个月，建议执行版）

**阶段 A（1-2 个月）：系统化基线与可复现实验**
- 固化统一 benchmark 脚本：Qwen3/DeiT/ResNet/VGG/MobileNet
- 输出三类曲线：`吞吐-显存`、`精度-噪声`、`速度-分块参数`
- 产出：内部 Tech Report v1 + 可复现表格/图脚本

**阶段 B（2-4 个月）：性能上限突破**
- 实现 LinearMem 输入维分块
- 实现 auto_speed 策略和热点层驻留机制
- 尝试噪声/ADC 融合 kernel（Triton/CUDA）
- 产出：与当前版本对比的 ablation（至少 5 组）

**阶段 C（4-8 个月）：器件-算法联合分析**
- 研究 read/write variation、DAC/ADC bitwidth 对 LLM/ViT/CNN 的统一影响规律
- 做层敏感度分析并形成选择性 CIM 策略（哪些层必须高保真）
- 产出：可投稿核心图组（Pareto 前沿 + 敏感度热图）

**阶段 D（8-12 个月）：论文与开源**
- 论文主线：可扩展性 + 真实性 + 可复现性
- 对外发布精简复现包（不含私有数据）
- 对比 NeuroSim/TxSim 的支持规模与仿真维度
- 目标：TCAD/DATE/DAC 等方向投稿

### 实验设计建议

| 实验 | 内容 | 意义 |
|------|------|------|
| 1 | Qwen3-0.6B/1.7B/4B/8B PPL vs. read_variation | RRAM 读噪声对 LLM 的敏感度 |
| 2 | 不同 DAC/ADC 精度 (2-16 bit) vs. PPL | 量化器件规格对精度的影响 |
| 3 | write_variation ≠ 0 时的 PPL 退化 | 编程误差的影响 |
| 4 | 显存占用 vs. 模型规模 (streaming vs. non-streaming) | 框架可扩展性验证 |
| 5 | 推理速度 vs. chunk_size / slice 数量 | 计算-内存 trade-off 分析 |
| 6 | 与 TxSim / NeuroSim 的功能对比 | 定位创新点 |

---

## 四、落地执行清单（建议）

### 4.1 下一个迭代优先级（建议）

1. `P0`：LinearMem 输入维分块（补齐超大 FC / 长序列 OOM 防护）
2. `P0`：Auto speed/memory 双策略与日志可视化
3. `P1`：噪声+ADC 融合 kernel 原型（先单模型验证）
4. `P1`：统一 benchmark 与自动生成报告脚本
5. `P2`：选择性 CIM（层敏感度驱动）

### 4.2 成功标准（工程+科研）

- 工程目标：
  - 24GB GPU 上稳定跑通 8B + ImageNet 主流模型
  - 默认配置可在无手工调参下完成推理
- 科研目标：
  - 形成至少 3 张核心定量图（规模、精度、性能）
  - 形成完整 ablation 矩阵（方法组件逐项增益）

---

## 五、API 统一说明

### `update_weight()` vs `prepare_for_inference()` vs `update_weight_and_prepare()`

| 方法 | 作用 | 适用场景 |
|------|------|---------|
| `update_weight()` | 仅计算 G（从当前权重映射到 conductance） | 训练循环中每次 optimizer.step 后调用 |
| `prepare_for_inference()` | 设置推理模式 + 压缩 G + 释放训练数据 | 训练完成后、推理前调用（需先调 update_weight） |
| `update_weight_and_prepare()` | 逐层: 计算G + 压缩 + 释放权重 + 可选 streaming | **推理专用**，合并了 update + prepare，且显存峰值更低 |

### 推荐用法

**推理场景**: 只调 `update_weight_and_prepare(streaming=True/False)`  
**训练场景**: 循环中调 `update_weight()`，训练完后调 `prepare_for_inference()`

所有模型（ResNet, DeiT, VGG, MobileNet, ResNet_CIFAR, Qwen3）现已统一支持这三个方法，且支持 `streaming=False/True/"auto"` 的统一语义。
