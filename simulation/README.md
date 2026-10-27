# 线损根因仿真数据集

当前正式数据由 `generate_pandapower_datasets.py` 生成。该脚本使用
`pandapower.networks.case33bw()` 建立 IEEE 33 节点系统，并通过
`pandapower.runpp()` 执行 Newton–Raphson 交流潮流。

## 输出

推荐将 v2 数据输出到 `data/simulated_pandapower_v2/`：

- `technical_line_loss.csv`：技术性线损及正常对照；
- `non_technical_line_loss.csv`：非技术性线损及正常对照；
- `mixed_line_loss.csv`：正常、单一技术性、单一非技术性和混合根因；
- `manifest.json`：生成器、随机种子和输入输出说明；
- `quality_report.json`：样本数、类别分布、缺失值、潮流收敛和能量平衡审计。

## v2生成逻辑

生成器在根因干预前加入与标签无关的节点负荷、无功、源端电压、线路电阻
和背景分布式电源波动；随后注入6类技术性根因、6类非技术性根因及其混合
组合。计量干预后加入测量噪声，并将表计总量投影至物理可行域，以满足：

`feeder_input_kw = metered_load_kw + technical_loss_kw + non_technical_loss_kw`

数据按时间顺序划分为约70%训练集、15%验证集和15%测试集。

## 复现

```powershell
C:\Users\MAO\anaconda3\envs\pytorch_flask\python.exe simulation\generate_pandapower_datasets.py --samples 6000 --output-dir data\simulated_pandapower_v2
```

默认随机种子为 `20260806`。生成器已内置计量投影逻辑；
`repair_pandapower_v2.py` 仅用于修复加入该逻辑之前生成的历史v2文件，重新生成
数据时无需再运行。

## 适用范围

该数据属于基于 pandapower、受经验负荷曲线约束的半合成基准，适合根因分类、
排序、多标签定位和消融实验，不应表述为真实馈线现场数据，也不能单独证明模型
具备工程部署能力。
