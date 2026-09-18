# V4：工程入口、配置与开发检查

V4 基于 V3 整理项目的运行和维护方式，保留 GPU 训练路线及 V3 的标签、模型、预处理和预测契约。当前仍是用于技术审阅的研究项目候选版，没有新增历史收益或模型性能结论。

## 交付内容

| 变更 | 作用 |
| --- | --- |
| 统一 Python 入口 | `scripts/run_pipeline.py` 负责阶段选择、依赖/GPU 预检和失败退出；`run_all.sh` 调用同一个入口。 |
| 配置预检 | 检查必要字段、日期切分、预测 horizon、运行名、模型路径、评估覆盖和路径占位符；重复 JSON 键及非有限数值会报错。 |
| 统一相对路径 | `project_root` 相对仓库解析，其余已登记路径相对项目根目录解析；主线、审计、回测及可视化入口共用加载器。 |
| 只预览运行计划 | `--dry-run` 只用 Python 标准库，不读取行情、不加载 TensorFlow、不创建产物目录。 |
| 消除隐式覆盖 | 标签缩放读取 `label.fixed_scale`；Bash 入口拒绝 `LABEL_SCALE`。修正 Stage1 的 `stage0.enable`/`enabled` 拼写不一致。 |
| 分层依赖 | 将已测试的 core/dev 环境与 GPU、历史实验候选依赖分开，并约束 core 的间接依赖版本。 |
| 本地与 CI 共用检查 | `scripts/check.py` 检查 Python/配置语法、依赖、Ruff 正确性规则和回归测试；GitHub workflow 使用同一命令。 |
| 维护文档 | 增加环境说明、开发流程、变更记录及本迁移说明，保留原阶段文件名以对应历史记录。 |

未把项目发布为 Python 安装包；`pyproject.toml` 用来配置开发工具。Stage0、LightGBM 和历史 tick 标签脚本仍在默认管线之外。

## V4 收尾修订

本次继续处理 V4 时，复现并修复了配置预检的遗漏：字符串开关可能被当作真值、关闭时间戳输出后训练无法加载窗口、不同阶段可能覆盖同一个统计或清单文件、异常回测校准字段可能直接触发 `TypeError`。这些配置现在会在依赖/GPU 检查和数据处理之前以明确的配置错误退出。

同时提前检查主线支持的优化器、标签裁剪和样本权重模式，以及相关数值范围。示例配置删除了训练代码从未读取的 `train.optimizer.eps`，继续保留原来实际生效的 `epsilon=1e-7`；旧私有配置需要同步移除 `eps`。本修订没有更改训练算法、模型结构或标签计算公式。

增加 8 项回归测试；还扩展了原有非有限 JSON 数值测试，覆盖 `1e999` 转换为无穷大的情况。允许的读写路径引用、关闭可选产物和复用既有输入统计等配置仍可通过预检。

## 验证结果

- 在独立的 Linux Python 3.12.14 虚拟环境安装 `requirements/dev.txt`，没有借用系统的 NumPy、pandas 或 SciPy。
- 64 项测试通过，其中 V3 原有 43 项，V4 配置/入口测试共 21 项（首次交付 13 项，收尾新增 8 项）。
- 新测试覆盖跨工作目录、带空格的配置路径、无第三方包 dry-run、配置缩放值、重复/异常配置、审计参数、GPU 预检失败及下游停止执行。
- `pip check`、Ruff 正确性检查、Python 编译、示例配置校验、Bash 语法与 Git 空白检查通过。
- GitHub workflow 的 YAML、触发条件和检查入口已在本地检查；还没有 GitHub 托管运行结果。
- GPU 候选依赖通过 pip 解析检查；这不代表安装、模型保存重载、GPU 计算或训练已经验证。

Ruff 采用针对正确性的规则，不表示旧实验代码已经完成全面风格重写。配置检查针对当前主线支持的字段，不是所有历史实验参数的完整 JSON Schema。

## 现在可以做的检查

从解压后的仓库根目录，先查看运行计划，无需 GPU 或第三方包：

```bash
python scripts/run_pipeline.py --config-path config/gp_lit_regression_v6_gpmain_64.example.json --dry-run
```

在独立 Python 3.12 环境安装开发依赖后运行完整本地检查：

```bash
python -m pip install -r requirements/dev.txt
python scripts/check.py
```

Linux/WSL 与 Windows PowerShell 的虚拟环境命令见 [environment.md](environment.md)。Windows 运行未在当前环境实测；训练目标仍是 Linux/WSL2 NVIDIA GPU。

## 从 V3 迁移

1. 将 V4 解压到新目录，保留 V3 和历史产物。
2. 从公开模板创建私有 `.local.json`，或迁移已有 V3 配置，然后运行配置校验。明确设置 `project_root`；相对值现在固定以仓库为基准。
3. 确认真实股票代码、输入根目录和日期已替换占位值；主线 Stage1 使用数字股票代码发现文件。单股票保持归一化 map 为 `null`。
4. 把以前的 `LABEL_SCALE` 值写入 `label.fixed_scale`，并移除该环境变量。若旧配置有 `stage0.enable`，改为 `stage0.enabled`。
5. `train`、`predict`、`backtest` 使用同一运行名；预测模型路径应指向 `paths.results_dir` 下的对应 Stage6 输出。
6. 删除 `train.optimizer.eps`，显式核对 `epsilon`；保持 `stage3.write_t_sec=true`。配置开关使用 JSON 布尔值，各阶段清单、统计和报告的输出路径应互不覆盖。更完整的检查范围见 [配置说明](../config/README.md)。
7. 旧产物仍受 V3 指纹契约约束。移动或重打包产物后，不要手工改写契约绕过校验，也不要覆盖归档实验。V4 不自动迁移模型。

`--stages` 只选择已有依赖关系中的部分阶段，不能恢复中断的训练，也不会自动生成前置产物。例如，只执行已具备输入文件的数据准备阶段：

```bash
python scripts/run_pipeline.py --config-path config/research.local.json --stages manifest labels pack windows healthcheck
```

包含 `train` 或 `predict` 的计划在开始前检查 GPU。任何阶段失败都会停止后续阶段。Stage0 不会因 `enabled=true` 自动运行；该开关只让 Stage1 读取已生成的 enriched 输入目录。

## 后续边界

V5 的下一项工作是完整合成数据 Stage1–5 集成流程及产物一致性验证。GPU 环境实测、TensorFlow 数值/序列化验证、真实数据重新训练和最终展示仍是后续事项。本版没有提交或发布 GitHub，也没有宣称 CI、GPU 训练或历史指标已复现。
