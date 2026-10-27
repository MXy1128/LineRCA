# Pandapower 线损根因分析数据集

这是以项目现有小时负荷序列为时间驱动、通过 pandapower 2.14.10 在 IEEE 33 节点配电网上逐快照执行 AC 潮流得到的半合成根因基准。

## 文件

- `technical_line_loss.csv`：技术性线损根因与正常对照。
- `non_technical_line_loss.csv`：非技术性线损根因与正常对照。
- `mixed_line_loss.csv`：技术、非技术、混合和正常四类样本。
- `manifest.json`：版本、网络、生成器与输入信息。
- `quality_report.json`：收敛、缺失、类别和能量平衡检查。

三套数据各有 5,384 行、241 列；训练/验证/测试按时间切分为 3,769/807/808 行。

## 标签字段

- `loss_type`：`normal`、`technical`、`non_technical` 或 `mixed`。
- `root_cause`：便于单标签实验的组合标签。
- `technical_root_cause`：技术根因；混合实验应使用该字段做第一个标签头。
- `ntl_root_cause`：非技术根因；混合实验应使用该字段做第二个标签头。
- `technical_root_element_type/index`：技术根因所在的线路、负荷、母线或外部电网。
- `ntl_root_bus`：非技术损耗所在母线。
- `technical_anomaly`、`ntl_anomaly`：两类二元异常标签。
- `technical_severity`、`ntl_severity`：干预强度。

## 特征组

- 系统级：实际负荷、表计负荷、馈线输入、P/Q、技术/非技术/总损耗、最低电压和最大线路负载率。
- 母线级：`bus_XX_vm_pu`、`bus_XX_actual_load_kw`、`bus_XX_metered_load_kw`。
- 线路级：`line_XX_in_service`、`line_XX_loss_kw`、`line_XX_loading_pct`。
- 时间级：时间戳、小时、工作日和数据切分。

## 物理与计量口径

技术损耗直接取 pandapower 潮流结果：`res_line.pl_mw + res_trafo.pl_mw`。IEEE 33 节点模型没有变压器表，因此当前变压器损耗为零。

非技术损耗定义为：

`non_technical_loss_kw = actual_load_kw - metered_load_kw`

每行满足：

`feeder_input_kw = metered_load_kw + technical_loss_kw + non_technical_loss_kw`

线路明细损耗之和与 `line_loss_kw` 最大差约 `6e-8 kW`（CSV 小数截断造成）；原始计算的最大能量平衡残差约 `2.2e-12 kW`。

## 根因注入

技术性：过载、线路电阻增加、源端低电压、低功率因数、分布式电源反向潮流、拓扑重构。

非技术性：均匀缩放、表计旁路、峰时窃电、均值替换、间歇掉线、系统性计量偏差。

根因按“电网区域×日期”连续注入，避免逐小时完全随机的标签闪烁。峰时窃电只在 17:00–22:00 激活，间歇掉线只在抽中的小时激活，因此这两类有效样本数较少，符合其事件机制。

## 推荐实验

1. 使用 `split` 做严格时间外推，不要随机重划分相邻样本。
2. 先预测 `loss_type`，再做技术根因、非技术根因和位置定位的多任务学习。
3. 输入模型时排除所有标签、根因位置和 severity 字段，防止目标泄漏。
4. 增加“整类根因留出”实验，检验 zero-day 根因泛化。
5. 论文中称为 *pandapower-based semi-synthetic dataset*，不要称为真实窃电数据。

## 复现

```powershell
$env:NUMBA_DISABLE_JIT='1'
C:\Users\MAO\anaconda3\envs\pytorch_flask\python.exe simulation\generate_pandapower_datasets.py --samples 6000
```

请求 6000 个等间隔时间点；剔除原始文件中所选电网负荷为空或非正的点后，有效样本为 5,384。

## 依据

- [pandapower line results](https://pandapower.readthedocs.io/en/latest/elements/line.html)
- [OpenDSS power-flow and loss outputs](https://opendss.epri.com/PowerFlow.html)
- [IEEE PES test feeders](https://ieee-pes-data-sharing.org/datasets/detail/9442e4ab-ea05-4042-baca-f4044f93c055)
- [Synthetic electricity-theft attacks, IET GTD](https://doi.org/10.1049/gtd2.12997)
- [Practical feature engineering for electricity-theft detection](https://doi.org/10.1016/j.apenergy.2019.01.047)
- [Limits of synthetic attack models](https://doi.org/10.1016/j.energy.2025.137763)
