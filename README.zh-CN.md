# 不规则事件时间下的 LOB 短期收益预测

**利用 Level-2 订单簿已有因子与真实事件时间，研究未来数秒的收益回归。**

[English](README.md) · [项目详解](docs/project_walkthrough.zh-CN.md) · [模型与管线架构](docs/architecture.md) · [验证证据](docs/evidence.md)

本项目研究近期盘口因子是否包含未来中间价收益的信息。模型采用带连续时间编码的 Transformer–LSTM，管线明确规定标签、时间切分、窗口采样、预测对齐和实验产物之间的关系。

**当前状态：**用于研究与求职展示。V7 整理展示材料，保留 V6 实现；已有合成数据与模型组件检查记录。当前版本的完整 GPU 实验、真实数据效果及历史结果复现仍待完成。公开包不含原始行情和训练好的研究模型。

## 研究任务

| 项目 | 公开配置示例 |
| --- | --- |
| 因子输入 | 已有的 109 个盘口相关因子，来自不等时间间隔的事件行 |
| 历史窗口 | 同一交易时段内连续 64 行，同时满足历史时间跨度约束 |
| 时间输入 | 每行相对于窗口首行的经过秒数 |
| 回归目标 | 未来 **2.5—3.5 秒**区间内，按事件行等权平均的参考价格相对当前价格的收益 |
| 模型 | 训练集标准化 → 投影与正弦时间编码相加 → Transformer → LSTM → 单一回归输出 |
| 训练 | TensorFlow GPU；默认加权 CCC，目前限制为单副本 |
| 验证 | 按日期顺序划分；依据无权重验证集 Pearson 相关系数选择模型 |

这些数字是配置选择，不是实验成绩。当前公开代码是**单目标回归**；原始数据已含因子列，本仓库不包含这些私有因子的完整定义。

![示意时间线：历史事件间隔不等，未来2.5至3.5秒仅用于标签构造。](docs/assets/event_time_target.svg)

*此图为概念示意，不是真实行情或回测结果。64 个事件覆盖的秒数会变化；未来区间只参与标签构造，不进入模型输入。*

## 项目中可以检查的设计

- **保留真实时间信息。** 先用 float64 时间做窗口内相减，再输入相对时间；时间编码不把事件行视为等间隔。混合精度检查覆盖毫秒差值保留。
- **明确标签口径。** 未来区间中的有效价格按事件等权平均；未来信息不足时标签无效，不补成零收益；标签缩放和裁剪参数有明确来源。
- **保持训练与推理一致。** 打包和原始 CSV 推理共用特征预处理。预测保存原始终点行号及时间戳，评估逐行核对。
- **防止实验产物误用。** 标签定义、缩放和统计指纹随数据包、模型与预测传递；相同 horizon 名称也不能掩盖不同标签定义。

详见[架构与设计取舍](docs/architecture.md)、[数据契约](docs/data_contract.md)和[推理与回测口径](docs/inference_and_backtest_contract.md)。

## 已有证据及适用范围

以下是 **V6 已记录的检查结果**，对应源码在本次文档版中保持不变。它们支持工程正确性判断，不能代替真实数据上的预测效果。

| 证据 | 检查内容 | 范围 |
| --- | --- | --- |
| [80 项非 TensorFlow 测试](docs/verification/v6_checks_summary.json) | 标签、时间精度、配置、打包、预测对齐、失败停止和回测规则 | 合成数据与指定回归用例 |
| [完整合成 Stage1—5 检查](docs/verification/v6_synthetic_summary.json) | 所有合成标签和有效窗口与独立参考实现对照；改变验证/测试数据不改变训练统计 | 每个场景 8 个合成交易时段，不测预测收益 |
| [13 项模型与集成检查](docs/verification/v6_model_components.json) | 损失与指标、两种精度策略、保存加载、预测生成与复用、评估 | CPU 小规模组件及临时模型用例，不是 Stage6 的 GPU 研究训练 |
| GPU 验收与托管 CI | 已提供命令和工作流 | GPU 成功路径与 GitHub 上的执行尚未验证 |

当前版本不宣称测试集 IC、方向准确率、交易收益或优于 LightGBM。各项证据与代码的对应关系见[证据说明](docs/evidence.md)。

## 快速查看代码

| 想了解的问题 | 入口 |
| --- | --- |
| 预测目标具体如何定义？ | [`src/data_contract.py`](src/data_contract.py) |
| 模型如何利用不规则时间？ | [`src/stage6_train_regression.py`](src/stage6_train_regression.py) |
| 怎样拒绝新旧标签混用？ | [`src/target_contract.py`](src/target_contract.py) |
| 打包与推理如何采用同一预处理？ | [`src/feature_preprocessing.py`](src/feature_preprocessing.py) |
| 如何独立验证标签和窗口？ | [`scripts/synthetic_data.py`](scripts/synthetic_data.py)、[`tests/test_synthetic_pipeline.py`](tests/test_synthetic_pipeline.py) |

[项目详解](docs/project_walkthrough.zh-CN.md)提供研究动机、关键取舍和具体修复案例。

## 本地查看与验证

在仓库根目录，使用 Python 3.12 查看管线计划，无需数据、TensorFlow 或 GPU：

```bash
python scripts/run_pipeline.py --config-path config/gp_lit_regression_v6_gpmain_64.example.json --dry-run
```

在独立开发环境中执行基础检查：

```bash
python -m pip install -r requirements/dev.txt
python scripts/check.py
```

若要保留一组可检查的合成数据产物，指定一个尚不存在的目录：

```bash
python scripts/check_synthetic_pipeline.py --output-dir local/synthetic-review
```

[环境说明](docs/environment.md)包含 Linux/WSL 和 PowerShell 命令；[模型检查说明](docs/model_verification.md)提供独立的 TensorFlow 检查方式。这些检查不构成 CPU 研究训练入口。

## 实际研究运行

主线为 **清单 → 标签 → 行级打包 → 窗口 → 健康检查 → GPU 训练 → 预测 → 评估**。入口是 `scripts/run_pipeline.py`，`run_all.sh` 调用同一入口；阶段失败后停止后续执行。

公开示例采用单股票、原有因子、关闭跨股票尺度归一化。单股票仍使用训练集特征均值和标准差。Stage0 新因子和 LightGBM 对照不在默认主线；Stage1.5 尺度审计和研究回测按需启用。

真实运行需要私有数据、独立的 `*.local.json`、GPU 环境和新的 run_name，步骤见[运行说明](docs/research_runbook.md)。旧版/V5 产物迁移时应重建 Stage2—5 并重新训练，不能通过修改版本字段或 `--only-scale` 升级旧模型。

## 研究局限与下一步

时间编码使模型能够利用实际经过时间，其预测价值仍需消融实验验证。按日期切分和训练集统计隔离处理了部分泄漏风险，外部已有因子自身是否因果仍需单独核实。

可选回测使用验证集标定阈值和买卖报价，但同一行产生信号并执行仍是理想化假设。代码未覆盖延迟、成交不确定性、队列位置与市场冲击；累计单笔收益也不等于按资金计算的净值曲线。

下一研究里程碑是完成干净环境下的 GPU 实验，再比较统一口径的基线、时间编码消融和不同时期/交易时段表现。历史数值在数据、标签、模型及评估口径核实前单独保存。

## 其他文档

[架构](docs/architecture.md) · [证据](docs/evidence.md) · [配置](config/README.md) · [运行](docs/research_runbook.md) · [环境](docs/environment.md) · [V7 更新说明](docs/v7_release_notes.zh-CN.md)

本仓库准备作为作品集供审阅，尚未添加开源许可证。公开包排除原始数据、训练产物和本地配置。
