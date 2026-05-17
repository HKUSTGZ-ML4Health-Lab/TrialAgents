# scripts 目录说明

本目录包含用于分析、实验和语料库构建的各种脚本文件。

## 目录结构

```
scripts/
├── README.md               # 本文件
├── analysis/               # 数据分析脚本 (从根目录迁移)
│   ├── agent_diversity.py      # <- agent_diversity_analyzer.py
│   ├── oncology_diversity.py   # <- oncology_agent_diversity_analyzer.py
│   └── case_analysis.py        # <- current_case_analysis.py
│
├── corpus/                 # 语料库构建脚本 (从corpus/corpus_construct/迁移)
│   ├── build_corpus.py
│   └── fix_msd_index.py        # <- fix_msd_index.py
│
├── experiments/            # 实验脚本 (从根目录迁移)
│   ├── multi_loop.py           # <- multi_loop.py
│   ├── syndata.py              # <- syndata.py
│   └── hyperparam_search.py    # <- hyperparameter_search.py
│
└── legacy/                 # 废弃/备份脚本 (从根目录迁移)
    ├── baselines_backup.py     # <- baselines_backup.py
    ├── framework_backup.py     # <- utils/framework_backup.py
    └── ...                     # 其他*_backup.py文件
```

## 脚本迁移映射表

### 从根目录迁移

| 原文件 | 新位置 | 说明 |
|--------|--------|------|
| `agent_diversity_analyzer.py` | `scripts/analysis/agent_diversity.py` | 智能体多样性分析 |
| `oncology_agent_diversity_analyzer.py` | `scripts/analysis/oncology_diversity.py` | 肿瘤科智能体多样性分析 |
| `current_case_analysis.py` | `scripts/analysis/case_analysis.py` | 当前案例分析 |
| `multi_loop.py` | `scripts/experiments/multi_loop.py` | 多轮实验 |
| `syndata.py` | `scripts/experiments/syndata.py` | 合成数据生成 |
| `hyperparameter_search.py` | `scripts/experiments/hyperparam_search.py` | 超参数搜索 |
| `fix_msd_index.py` | `scripts/corpus/fix_msd_index.py` | MSD索引修复 |
| `baselines_backup.py` | `scripts/legacy/baselines_backup.py` | 基线备份 |

### 从corpus/corpus_construct/迁移

| 原文件 | 新位置 | 说明 |
|--------|--------|------|
| `corpus/corpus_construct/*` | `scripts/corpus/` | 语料库构建脚本 |

### 从utils/迁移

| 原文件 | 新位置 | 说明 |
|--------|--------|------|
| `utils/framework_backup.py` | `scripts/legacy/framework_backup.py` | 框架备份 |
| `utils/framework_enhanced.py` | `scripts/legacy/framework_enhanced.py` | 增强框架备份 |

## 使用示例

### 运行分析脚本

```bash
cd scripts/analysis
python agent_diversity.py --dataset NCT00694382
```

### 运行实验脚本

```bash
cd scripts/experiments
python multi_loop.py --config configs/experiment.yaml
```

### 构建语料库

```bash
cd scripts/corpus
python build_corpus.py --source ctgov --output ../../data/raw/ctgov/
```

## 注意事项

1. 迁移时需保持文件功能不变
2. 更新脚本中的相对路径引用
3. 确保依赖包在脚本开头正确导入
4. 建议在迁移后对每个脚本进行测试

## 主入口脚本保留在根目录

以下脚本作为项目主入口，保留在根目录：

- `collaboration_pipeline.py` - 完整的协作流程（未来可迁移到src/pipeline/main_pipeline.py）
- `main.py` (建议新建) - 项目主入口
- `run_mortality_prediction.py` (建议新建) - 死亡率预测任务入口
- `run_digital_twin.py` (建议新建) - 数字孪生任务入口
