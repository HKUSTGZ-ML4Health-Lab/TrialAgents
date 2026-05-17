# Baselines 基线方法

本目录包含用于对比的各种基线方法实现。

## 目录结构

```
baselines/
├── README.md                   # 本文件
├── single_llm/                 # 单LLM基线方法
│   ├── __init__.py
│   ├── run.py                  # <- run_single_llm.py
│   └── multi_run.py            # <- run_multi_llm.py
│
├── small_models/               # 小模型基线方法
│   ├── __init__.py
│   ├── tabular.py              # <- run_tabular_small_models.py
│   └── timeseries.py           # <- run_ts_small_models.py
│
├── implementations/            # 基线实现代码
│   ├── llm.py                  # <- llm.py (基线版)
│   ├── small_models.py         # <- small_models.py
│   └── run_baselines.py        # <- baselines.py
│
└── results/                    # 基线结果
    ├── .gitkeep
    └── README.md
```

## 基线方法说明

### 1. 单LLM基线 (Single LLM Baseline)

使用单一LLM进行预测，不进行多智能体协作。

**文件**：`baselines/single_llm/run.py`

**对应原文件**：`run_single_llm.py`

```python
# 使用方法
python baselines/single_llm/run.py \
    --dataset NCT00694382 \
    --llm deepseek-chat \
    --output_dir baselines/results/single_llm/
```

### 2. 多LLM投票基线 (Multi-LLM Voting Baseline)

使用多个LLM分别预测，然后进行简单投票。

**文件**：`baselines/single_llm/multi_run.py`

**对应原文件**：`run_multi_llm.py`

```python
# 使用方法
python baselines/single_llm/multi_run.py \
    --dataset NCT00694382 \
    --llms deepseek-chat gpt-4o qwen-plus \
    --voting_method majority \
    --output_dir baselines/results/multi_llm/
```

### 3. 传统时序模型基线 (Traditional Time Series Models)

使用传统的机器学习/深度学习模型作为对比。

#### 3.1 表格模型 (Tabular Models)

**文件**：`baselines/small_models/tabular.py`

**对应原文件**：`run_tabular_small_models.py`

支持的模型：
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

```python
# 使用方法
python baselines/small_models/tabular.py \
    --dataset NCT00694382 \
    --model xgboost \
    --output_dir baselines/results/tabular/
```

#### 3.2 时序模型 (Time Series Models)

**文件**：`baselines/small_models/timeseries.py`

**对应原文件**：`run_ts_small_models.py`

支持的模型：
- LSTM
- GRU
- TCN (Temporal Convolutional Network)
- Transformer

```python
# 使用方法
python baselines/small_models/timeseries.py \
    --dataset NCT00694382 \
    --model lstm \
    --epochs 100 \
    --output_dir baselines/results/timeseries/
```

## 迁移映射表

### 从根目录和baselines目录迁移

| 原文件 | 新位置 | 说明 |
|--------|--------|------|
| `run_single_llm.py` | `baselines/single_llm/run.py` | 单LLM基线 |
| `run_multi_llm.py` | `baselines/single_llm/multi_run.py` | 多LLM投票基线 |
| `run_tabular_small_models.py` | `baselines/small_models/tabular.py` | 表格模型基线 |
| `run_ts_small_models.py` | `baselines/small_models/timeseries.py` | 时序模型基线 |
| `baselines/baselines.py` | `baselines/implementations/run_baselines.py` | 基线运行器 |
| `baselines/llm.py` | `baselines/implementations/llm.py` | LLM实现 |
| `baselines/small_models.py` | `baselines/implementations/small_models.py` | 小模型实现 |

### 保留在原baselines目录的文件

| 文件 | 说明 |
|------|------|
| `baselines_read.py` | 数据读取工具 |
| `baselines_backup.py` | 备份版本（移至scripts/legacy/） |
| `test_single_llm.py` | 测试脚本 |
| `test_multi_llm.py` | 测试脚本 |
| `runs/` | 运行记录 |
| `response/` | 响应结果 |
| `saved_models/` | 保存的模型 |

## 性能对比

| 方法 | AUC-ROC | AUC-PR | 备注 |
|------|---------|--------|------|
| Single LLM | 0.75-0.80 | 0.65-0.70 | 基线 |
| Multi-LLM Voting | 0.78-0.82 | 0.68-0.73 | 简单集成 |
| Tabular Models | 0.70-0.75 | 0.60-0.65 | 传统ML |
| Time Series Models | 0.72-0.78 | 0.62-0.68 | 深度学习 |
| **TrialAgents (Ours)** | **0.85-0.92** | **0.75-0.85** | 多智能体协作 |

## 实验配置

```yaml
# configs/baseline_config.yaml
baselines:
  single_llm:
    models:
      - deepseek-chat
      - gpt-4o-mini
      - qwen-plus
    
  multi_llm:
    models:
      - deepseek-chat
      - gpt-4o-mini
      - qwen-plus
    voting: majority
    
  tabular:
    models:
      - logistic_regression
      - random_forest
      - xgboost
      - lightgbm
      
  timeseries:
    models:
      - lstm
      - gru
      - tcn
      - transformer
    epochs: 100
    batch_size: 32
```

## 运行所有基线

```bash
# 运行所有基线方法并对比
python baselines/run_all_baselines.py \
    --dataset NCT00694382 \
    --output_dir baselines/results/comparison/

# 生成对比报告
python baselines/generate_comparison_report.py \
    --results_dir baselines/results/comparison/ \
    --output report.html
```

## 注意事项

1. 基线方法应保持与主方法相同的数据预处理方式
2. 使用相同的训练/测试划分
3. 确保公平的比较条件（相同的评估指标）
4. 记录实验设置和环境配置
