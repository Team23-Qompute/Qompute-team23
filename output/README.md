# Nutrition QAOA 结果图

这四张图来自一次完整的 16-qubit、CPU statevector 模拟：4 个 QAOA 深度 × 2 种优化器 × 3 个随机种子 × 3 次重启，共 72 次优化。每次最多调用目标函数 400 次。图表是对这次运行的描述，不代表真实量子硬件性能或临床饮食建议。

| 图片 | 看什么 |
| --- | --- |
| [depth_tradeoffs.png](depth_tradeoffs.png) | 真实期望成本、采到真实前 1% 菜单的概率，以及优化器是否在预算内收敛。误差棒是 **3 个 seed 平均值之间的标准差**；每个 seed 的值先对 3 次重启取平均。 |
| [paired_seed_variability.png](paired_seed_variability.png) | 同一 seed 下 p=2 与 p=4 的对比。p=2 平均成本较低主要受 seed 1 影响；不能把它解释成深度 2 固有地更好。 |
| [sampling_vs_uniform.png](sampling_vs_uniform.png) | 每次运行抽取 1,024／4,096／16,384 个模拟样本后，至少见到一次精确最优解的比例。橙线是在 65,536 个菜单中只有 1 个精确最优时，均匀随机独立抽样的理论概率。三档样本数使用同一随机序列的前缀，因此不是互相独立的试验。 |
| [surrogate_quality.png](surrogate_quality.png) | QUBO 近似在全体菜单上的高相关度，掩盖了最优附近较差的排序能力。两种加权拟合仅用于诊断，并未用于这次 QAOA。 |

这些图的数值输入保存在 [analysis_data.json](analysis_data.json)：只包含白名单中的汇总指标、seed 编号和采样计数。原始 notebook、日志、服务器环境信息和归档不随这些图提交。尤其要注意，“抽样找到最优”指从多个模拟样本中按**真实目标函数**评估后发现它；它不等于单次测量有很高的最优概率，也不证明相对穷举有运行时间优势。

重画现有图需要 Python、NumPy 和 Matplotlib：

```bash
python output/build_analysis.py
```

如果有新的完整结果目录，可以先刷新去标识化汇总数据，再重画：

```bash
python output/build_analysis.py --run-dir <completed-run-directory> --exact-optima 1
```

`--exact-optima` 必须是该运行中真实目标函数的精确最优菜单数量。本次使用严格数值容差复核为 1；其他数据集不能直接沿用。
