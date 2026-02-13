"""# MemIntelli 性能问题深度分析

## 一、问题概述

当前 MemIntelli 框架在推理场景下存在两个严重问题：
1. **速度极慢**：bit slicing 的计算量本身较大，加上为兼容训练保留的大量冗余操作，导致推理速度远低于预期。
2. **显存爆炸**：中间张量维度膨胀严重，多份数据副本同时驻留显存，ResNet18 在 ImageNet 上推理已接近显存上限。

---

## 二、整体数据流分析

以 `Conv2dMem.forward()` 为例，一次前向传播的数据流如下：

```
原始输入 (N, C, H, W)
  |
  +-- F.unfold() -> (N, C*kH*kW, L) -> transpose -> (N, L, C*kH*kW)
  |
  +-- SlicedData 创建（每次 forward 新建对象）
  |    +-- _slice_data() -> padding -> reshape -> quant_map_tensor()
  |    |    +-- sliced_data: (batch, num_div_row, num_div_col, num_slice, m, n)
  |    |    +-- quantized_data: (batch, rows, cols)  <-- 用于反向传播
  |    |    +-- max_data: (batch, num_div_row, num_div_col, 1, 1)
  |    +-- slice_data_imp() 完成
  |
  +-- engine.MapReduceDot(input_sliced, weight_sliced)
  |    +-- _num2V(input) -> Vin: 同 sliced_data 形状 + 噪声张量
  |    +-- _gen_read_noise(weight) -> G: 同 sliced_data 形状 + 多个临时张量
  |    +-- dot_high_dim(Vin, G) -> einsum -> 7-8维输出张量
  |    +-- ADC 量化 + 移位加权 + 缩放
  |    +-- sum + reshape -> 最终结果
  |
  +-- Conv2dMemRunc.forward()
       +-- ctx.save_for_backward(input_quantized, weight_quantized, ...)
       +-- F.fold() -> 输出 (N, C_out, H_out, W_out)
```

---

## 三、速度瓶颈详细分析

### 3.1 每次 forward 都重新切片输入（最大瓶颈之一）

**文件**: `NN_layers/linear.py` L52-54, `NN_layers/convolution.py` L83-89

```python
# LinearMem.forward() -- 每次调用都新建 SlicedData 并切片
def forward(self, input):
    input_sliced = SlicedData(self.input_slice_method, ...)   # 新建对象
    input_sliced.slice_data_imp(self.engine, input.detach())  # 完整切片流程
    return linear_mem_func(...)
```

`slice_data_imp()` 内部执行了以下昂贵操作：
- 矩阵 padding 到 `paral_size` 的整数倍
- 多次 reshape/transpose（7次以上维度变换）
- `quant_map_tensor()`：量化 + 位切片提取（含循环）
- 所有中间结果都分配新的显存

**影响**：对于 ResNet18，20个卷积层 x 每个 batch 都重复这套流程。

### 3.2 dot_high_dim 的 einsum 维度爆炸

**文件**: `pimpy/utils.py` L25-27

```python
# 带 batch 的情况
torch.einsum("bnmijk, mpskl->bnmpisjl", x, y)
```

以 `input_slice=(1,1,2,2)` 和 `weight_slice=(1,1,2,2)` 为例（各4个切片）：
- 输入 V: `(batch, num_div_row_x, num_div_col_x, 4, m, n)`
- 权重 G: `(num_div_row_y, num_div_col_y, 4, n, p)`
- **输出**: `(batch, num_div_row_x, num_div_col_y, 4, 4, m, p)` -- 切片维度的笛卡尔积

这意味着 **一次矩阵乘法被拆分成了 4x4=16 次子矩阵乘法**，且结果存储在一个联合张量中。切片越多（如 `(1,1,2,4)` 有4个切片），膨胀越严重。

### 3.3 _gen_read_noise() 实现效率极低

**文件**: `pimpy/memmat_tensor.py` L140-170

```python
def _gen_read_noise(self, mat):
    G = mat.G
    expanded_G = G.unsqueeze(-1)                         # 增加一个维度
    distances = torch.abs(expanded_G - expanded_levels)  # 全量距离矩阵
    closest_level_idx = torch.argmin(distances, dim=-1)  # argmin
    for level_idx, variation in self.read_variation.items():
        level_mask = (closest_level_idx == level_idx)
        level_noise = torch.normal(0, variation, G.shape, device=G.device)
        noise[level_mask] = level_noise[level_mask]
```

**问题**：
- `expanded_G` 在最后增加一个大小为 `g_level` 的维度，创建巨大的距离矩阵
- `torch.argmin` 在高维张量上运算慢
- `for` 循环遍历每个电导级别，每次创建完整大小的噪声张量
- 布尔索引 `noise[level_mask]` 在 GPU 上效率低下

### 3.4 _num2V() 每次创建噪声张量

**文件**: `pimpy/memmat_tensor.py` L172-186

```python
V_in = self.vread * torch.round(x.sliced_data / xmax * (self.rdac - 1)) / (self.rdac - 1)
V_in = V_in * (1 + torch.normal(0, self.vnoise, V_in.shape, device=V_in.device))
```

即使 `vnoise=0`，仍然会分配一个与 `V_in` 同形状的噪声张量。

### 3.5 _dot() 中大量冗余计算

**文件**: `pimpy/memmat_tensor.py` L188-260

`_dot()` 中存在：
- `G - self.LGS`：复制整个 G 张量
- ADC 量化：`torch.round(out / adcRef * (self.radc - 1))` 创建临时张量
- 两次 `torch.mul` + reshape + sum：多次中间张量分配
- 2D 和 3D 分支的代码近乎完全重复

### 3.6 Autograd Function 在推理时的开销

**文件**: `NN_layers/functions.py`

`LinearMemRunc.forward()` 和 `Conv2dMemRunc.forward()` 使用 `ctx.save_for_backward()`：

```python
ctx.save_for_backward(input_slice.quantized_data, weight_slice.quantized_data, bias)
```

虽然在 `torch.no_grad()` 下不会真正构建计算图，但：
- 代码仍然传递 `input` 和 `weight` 原始张量作为参数（即使推理不需要）
- `input.detach()` 的调用增加了不必要的操作
- 整个 autograd Function 的调度机制本身有额外开销

### 3.7 quant_map_tensor() 中的位提取循环

**文件**: `pimpy/utils.py` L63-66

```python
for idx in range(len(blk)):
    data_int[:, :, :, idx, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - idx])) >> b
    b += blk[-1 - idx]
```

每个切片都执行一次完整的张量运算（减法、取模、右移），且每次都创建临时张量。

---

## 四、显存瓶颈详细分析

### 4.1 SlicedData 存储多份数据副本

一个 `SlicedData` 对象同时持有：

| 属性 | 形状 | 用途 | 推理是否必需 |
|------|------|------|-------------|
| `sliced_data` | `(num_div_row, num_div_col, num_slice, m, n)` | 切片后的数据 | 权重不需要（已转为G），输入需要 |
| `quantized_data` | `(rows, cols)` | 量化后的数据 | **不需要**（仅反向传播用） |
| `max_data` | `(num_div_row, num_div_col, 1, 1)` | 每块最大值 | 需要 |
| `e_bias` | `(num_div_row, num_div_col, 1, 1)` | BFP 指数偏置 | 仅 BFP 模式需要 |
| `G` | `(num_div_row, num_div_col, num_slice, m, n)` | 电导值 | 需要（权重） |

**关键发现**：`slice_data_imp()` 中虽然在 `inference=True` 时会清除 `quantized_data` 和 `sliced_data`：

```python
if self.inference:
    self.quantized_data = None
    self.sliced_data = None
```

但是 **当前代码并未设置 `inference=True`**！ `LinearMem` 和 `Conv2dMem` 创建 `SlicedData` 时未传入 `inference=True` 参数：

```python
# LinearMem.__init__() -- 权重切片
self.weight_sliced = SlicedData(...)  # inference 默认为 False

# LinearMem.forward() -- 输入切片
input_sliced = SlicedData(...)  # inference 默认为 False
```

这意味着权重的 `sliced_data` 和 `quantized_data` 始终驻留在显存中，即使它们在推理时完全不需要。

### 4.2 维度膨胀的定量分析

以 ResNet18 第一个卷积层 `Conv2d(3, 64, 7x7, stride=2, padding=3)` 为例，输入 `(8, 3, 224, 224)`：

**原始权重**：`(64, 3, 7, 7)` = 9,408 参数

经过 unfold + 切片后（`weight_paral_size=(32,32)`, `weight_quant_gran=(64,64)`, 4个切片）：
- 转置后权重矩阵：`(147, 64)` -> padding 到 `(160, 64)` 
- 分块：`(5, 2, 32, 32)` -> 切片后 `sliced_data`: `(5, 2, 4, 32, 32)` = 40,960 元素
- `G` 张量：同形状 = 40,960 元素
- `quantized_data`：`(147, 64)` = 9,408 元素

**单层额外显存**：约 `(40960 + 40960 + 9408) x 4 bytes = 356 KB`

看似不多，但对于 **输入数据**（每次 forward 都创建）：
- unfold 后：`(8, 12544, 147)` 约 57 MB (float32)
- padding + 切片后 sliced_data: `(8, 392, 2, 4, 32, 32)` 约 308 MB
- quantized_data: 同原始大小 约 57 MB
- max_data 等辅助张量

**单层输入峰值显存 > 400 MB！**

### 4.3 dot_high_dim 的输出张量膨胀

继续上例，einsum 输出形状：
`(8, 392, 2, 4, 4, 32, 32)` = 每个元素 4 bytes -> **约 1.23 GB**

这只是一个层的一次中间计算结果。后续的 ADC 量化、乘法、sum 过程还会产生更多同等规模的临时张量。

### 4.4 Autograd 保存的张量

`Conv2dMemRunc.forward()` 中：

```python
ctx.save_for_backward(input_sliced.quantized_data, weight_sliced.quantized_data, bias,
                      torch.tensor(input.shape), torch.tensor(weight.shape))
```

在训练模式下，PyTorch 会保留这些张量直到反向传播完成。即使使用了 `torch.no_grad()`，autograd Function 的 forward 仍然会执行 `ctx.save_for_backward`，造成不必要的显存分配。

### 4.5 Conv2d 的 unfold 膨胀

`F.unfold` 将每个滑动窗口展开为向量：

| 层 | kernel | stride | 输入形状 | unfold 后形状 | 膨胀倍数 |
|----|--------|--------|----------|--------------|---------|
| conv1 (7x7) | 7 | 2 | (8,3,224,224) | (8,12544,147) | ~9.8x |
| layer1 conv (3x3) | 3 | 1 | (8,64,56,56) | (8,3136,576) | ~9x |
| layer2 conv (3x3) | 3 | 1 | (8,128,28,28) | (8,784,1152) | ~9x |
| layer3 conv (3x3) | 3 | 1 | (8,256,14,14) | (8,196,2304) | ~9x |
| layer4 conv (3x3) | 3 | 1 | (8,512,7,7) | (8,49,4608) | ~9x |

3x3 卷积核的 unfold 导致输入数据约膨胀 9 倍，之后还要经过切片进一步膨胀。

---

## 五、可行的优化方案

### 方案 A：推理专用快速路径（短期，改动小）

**目标**：不改变核心算法，仅消除推理时不必要的开销。

| 优化项 | 预计效果 | 难度 |
|--------|---------|------|
| 1. 启用 `inference=True` 清除权重的 `sliced_data`/`quantized_data` | 显存减少 ~30% | 低 |
| 2. 推理路径绕过 autograd Function，直接调用 `engine.MapReduceDot()` | 速度提升 ~10-20%，显存减少 | 低 |
| 3. `vnoise=0` 时跳过噪声张量分配 | 小幅速度/显存改善 | 低 |
| 4. 统一 `_gen_read_noise` 避免 per-level 循环，改用向量化 | 速度提升明显 | 中 |
| 5. `_dot()` 合并 2D/3D 代码路径 | 代码清洁，微小速度改善 | 低 |

### 方案 B：分块流水线处理（中期，核心优化）

**核心思想**：不一次性计算所有切片的笛卡尔积，而是逐块计算并累加。

```python
# 当前方式：一次性 einsum 产生 (batch, divs, 4, 4, m, p) 的巨大张量
result = torch.einsum("bnmijk, mpskl->bnmpisjl", V, G)

# 优化方式：逐切片对计算并原地累加
result = torch.zeros(batch, num_div_row_x, num_div_col_y, m, p)
for i in range(num_slices_x):
    for j in range(num_slices_y):
        partial = torch.einsum("bnmik, mpkl->bnmil", V[..., i, :, :], G[..., j, :, :])
        result += partial * shift_weights[i, j]
```

**优点**：峰值显存从 O(S_x * S_y) 降到 O(1)（S 为切片数）

**缺点**：循环可能略慢（但可以用 `torch.baddbmm` 等优化）

### 方案 C：权重预计算 + 缓存（中期）

权重在推理时是固定的，以下内容可以预计算并缓存：
- `G - self.LGS`：权重的净电导
- `shift_weights` 矩阵
- ADC 参考值 `adcRef`
- 权重侧的缩放系数

将这些合并为一个预处理过的权重张量，前向传播时只需要处理输入侧的切片和一次矩阵乘法。

### 方案 D：融合 kernel / 减少中间分配（中期-长期）

1. **融合 DAC-乘法-ADC**：将 `_num2V` -> `dot_high_dim` -> ADC量化 融合为一个操作，减少中间张量
2. **使用 `torch.compile`**：PyTorch 2.x 的编译器可以自动融合操作、减少内存分配
3. **原地操作**：尽可能使用 `out=` 参数或 `.mul_()` 等原地方法

### 方案 E：重写推理引擎（长期，效果最大）

创建一个独立的 `InferenceEngine`，完全不依赖 autograd：

```python
class InferenceEngine:
    def __init__(self, engine):
        self.engine = engine
    def prepare_weights(self, model):
        pass
    def forward_linear(self, input_data, prepared_weight):
        pass
```

**关键设计原则**：
- 不使用 `torch.autograd.Function`
- 不保存 `quantized_data`
- 权重的所有静态计算只做一次
- 输入切片尽可能精简（直接输出 V，跳过中间态）
- 支持半精度 (float16) 计算

### 方案 F：精度-性能权衡

| 策略 | 速度提升 | 显存节省 | 精度影响 |
|------|---------|---------|---------|
| float32 -> float16 | ~1.5-2x | ~50% | 轻微 |
| 减少切片数（如 4->2 个切片） | ~4x | ~4x | 需评估 |
| 增大 paral_size | 减少分块数 | 减少分块开销 | 无 |
| 增大 quant_gran | 减少量化组数 | 减少 max_data 大小 | 需评估 |

---

## 六、优先级建议

### 立即可做（1-2天）
1. **启用 `inference=True`**：在 `LinearMem` 和 `Conv2dMem` 中为权重的 `SlicedData` 启用 inference 标志
2. **推理绕过 autograd**：检测 `torch.no_grad()` 环境，直接调用 engine 而不经过 `LinearMemRunc`/`Conv2dMemRunc`
3. **条件噪声分配**：`vnoise=0` 时跳过噪声生成；所有 `read_variation` 为 0 时跳过 `_gen_read_noise`

### 短期（1周）
4. **分块流水线**：重写 `_dot()` 为逐切片对累加模式
5. **优化 `_gen_read_noise`**：用向量化替代 per-level 循环
6. **权重预计算**：将静态缩放系数合并到 G 中

### 中期（2-4周）
7. **独立推理引擎**：完全绕过训练代码路径
8. **`torch.compile` 集成**
9. **float16 支持**

---

## 七、预期效果

| 优化阶段 | 速度提升（预估） | 显存节省（预估） |
|----------|----------------|----------------|
| 立即可做 | 1.3-1.5x | 30-40% |
| 短期优化 | 3-5x | 50-70% |
| 中期重写 | 5-10x | 70-85% |

以 ResNet18 + ImageNet 为参考基准，优化后应能：
- batch_size 从 8 提升到 32-64
- 推理速度达到可用水平
- 为更大模型（ResNet50、MobileNetV2）提供可能性"""

import os
path = r"D:\OneDrive\code\MemIntelli_v2_20260212\analysis_performance_issues.md"
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Done! Wrote " + str(len(content)) + " chars to " + path)
'@
[IO.File]::WriteAllBytes("D:\OneDrive\code\_gen_md.py", [Text.Encoding]::UTF8.GetBytes($s))
python "D:\OneDrive\code\_gen_md.py"