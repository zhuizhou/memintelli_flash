# Mode1 差分对执行加速与 v2.1 发布设计

## 目标

以当前 DAC 投稿版本为唯一代码基线，将 mode0 中已经验证的紧凑电导恢复、tiled VMM/ADC和direct-final执行方法用于无slice展开的mode1差分对模式。首先在 `read_var=0.0` 下验证语义，再测量端到端加速比，最后把经过清理的代码发布到 GitHub `v2.1` 分支。

## 范围

- 主要优化对象是 `mode1`。该模式不展开input slice或weight slice；每个量化权重映射为一对 `G+`/`G-`，signed DAC输入驱动差分电导，差分电流在ADC前形成，ADC按input tile执行。
- `mode2` 只做兼容性回归，不作为本轮优化重点或主要性能结果。
- 不把论文实验数据、S1 planner、motivation/evaluation脚本、Nsight结果或临时诊断代码发布到 `v2.1`。

## 执行设计

### 语义参考路径

同一份 DAC 代码提供mode1 reference配置，关闭mode1 Triton G-index direct-final、input-tile grouping和chunked direct-final，使用原有PyTorch逐output tile、input tile执行路径。VMM、电压和电导使用BF16，ADC、tile scale和最终reduction使用FP32。

### 优化路径

第一阶段复用并整理 DAC 版本已有的mode1机制：

1. 从compact `G+`/`G-` level index直接执行差分VMM，不预先展开完整浮点电导张量。
2. 在同一Triton执行中完成signed DAC、`G+ - G-`电流、per-input-tile ADC、tile scale和最终输出累加。
3. 对宽输出按output tile chunk直接写最终二维Linear输出，避免框架级tile中间结果。
4. 通过input-tile grouping减少launch和atomic累加次数。

第二阶段先用单层profile确认现有direct-final的剩余瓶颈，再从下列候选中选择有数据支撑的最小优化，不预先承诺全部实现：

- activation voltage reuse：signed DAC只计算一次，避免同一输入随output tile重复量化。
- zero-variation differential index：`read_var=0.0`时将 `G+`/`G-`压缩成单个有符号差分level，减少一半索引读取；带read variation时仍保留双支路。
- atomic-free grouped reduction：若atomic累加占主导，改为受控partial reduction或单program tile reduction，同时限制额外working set。
- mode1 shape-aware schedule：依据tokens、input tiles、output tiles选择block和group，替代只适合少数shape的固定值。

任何新增优化必须通过独立开关关闭，以便构造同版本reference和性能消融。若mode1 fast path未命中，benchmark必须报告fallback计数并判为失败，不能用静默回退结果计算加速比。

## 正确性标准

### `read_var=0.0`

- 单层测试覆盖Q/K/V、gate/up、down和lm_head代表形状。
- 对比 reference 与优化路径的最终输出。
- FP32控制路径要求严格一致；BF16路径允许有限舍入误差，报告 `max_abs`、`mean_abs`、`max_rel`、cosine similarity 和 `torch_equal`。
- BF16默认验收阈值为 `allclose(rtol=1e-2, atol=1e-2)`，同时要求无 NaN/Inf。若输出量级表明该阈值过宽，测试改用基于reference幅值的更严格阈值。

### `read_var=0.05`

本轮不要求跨运行逐元素一致。完成 `read_var=0.0` 验收后，只做mode1功能和分布性检查：输出有限、均值/标准差合理、端到端可运行。随机噪声状态不同不能被解释为语义错误。

## 性能验证

- 远端 Pro6000优先使用空闲GPU。
- 正式口径：batch=1、seq=128、warmup=1、repeat=10、full Linear replacement including lm_head。
- 先跑单层热点形状，定位 kernel命中和瓶颈；再跑至少 Qwen 0.8B 和 4B 全模型。资源允许时补 9B。
- reference和优化路径必须使用同一模型、dtype、slice、array、ADC/DAC及GPU。
- 报告 mean/min latency、CUDA peak memory、优化路径命中计数和加速比 `reference_mean / optimized_mean`。
- 为排除GPU干扰，正式点至少运行三个独立进程；明显受占用干扰的运行废弃并重跑。

## 发布设计

1. 从 GitHub `v2.1` 创建独立干净工作区。
2. 以 DAC 投稿版本的运行时源码为准，白名单同步 package、必要测试、一个差分对benchmark和简洁使用文档。
3. 不复制研究目录、结果JSON、论文图表、服务器脚本和缓存。
4. 在发布树执行语法检查、CPU测试、GPU正确性和端到端benchmark。
5. 审计 `git status`、`git diff --stat`、大文件和敏感信息。
6. 以普通提交更新并推送 `origin/v2.1`；除非远端历史发生冲突，不使用强制推送。

## 验收条件

- DAC mode0现有测试与性能路径没有回归。
- mode1 `read_var=0.0` BF16输出满足误差阈值。
- mode2现有路径通过兼容性回归测试。
- mode1优化路径真实命中，且端到端速度优于同版本reference。
- 发布树只包含干净代码、测试、必要benchmark和文档。
- 推送后重新克隆/拉取 `origin/v2.1`，重复运行最小正确性测试。
