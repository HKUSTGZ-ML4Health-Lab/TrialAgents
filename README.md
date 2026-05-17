# TrialAgents

**Zero-Shot Multi-Agent Framework for Interpretable In-Silico Clinical Trial Rehearsal**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TrialAgents simulates a multidisciplinary clinical team with four LLM-driven agents to predict outcomes, generate privacy-preserving digital twins, and rehearse trial decisions without task-specific fine-tuning.

## Why TrialAgents

- **Beyond siloed predictions**: Models collaborative, evidence-based reasoning across specialties.
- **Interpretable by design**: Produces round-level reports, confidence signals, and consensus traces.
- **Privacy-preserving simulation**: Generates digital twins while reducing disclosure risks.
- **Model-agnostic deployment**: Works with DeepSeek, GPT, Gemini, and Qwen families.

## Core Architecture

TrialAgents follows a **4-agent, 3-stage workflow**:

### Agents

- **Clinician x3**: Oncology, Emergency Medicine, and Comprehensive Care specialists.
- **Supervisor (Unified)**: Combines Trialist (summary synthesis) and Supervisor (consensus orchestration, DPT triggering).
- **Simulator**: Autoregressively generates personalized and counterfactual patient trajectories.

### Stage 1 - Evidence-Based Review

Each Clinician retrieves domain evidence (via MedCPT) and produces an independent initial assessment.

### Stage 2 - Multi-Round Collaborative Diagnosis

The Supervisor coordinates iterative discussions, tracks disagreements, and drives consensus formation across rounds.

### Stage 3 - Task Completion

| Task | Component | Description |
|---|---|---|
| Mortality Prediction | DPT (LNL pipeline) | Refines noisy zero-shot outputs with lightweight temporal modeling |
| Digital Twin Generation | Simulator | Produces autoregressive, privacy-preserving patient trajectories |

## Key Mechanisms

### Diagnostic Predictive Tool (DPT)

- Uses high-confidence consensus outputs as pseudo-labels.
- Applies DivideMix (Learning with Noisy Labels) to train a lightweight Transformer.
- Improves robustness when ground-truth labels are sparse or unavailable.

### Self-Reflective Memory

- Archives historical diagnostic reports.
- Retrieves top-k similar cases to reduce repeated reasoning failures.

### Model-Agnostic Coordination

- Supports homogeneous or heterogeneous LLM configurations.
- Empirically, homogeneous settings can outperform mixed-model teams in this domain.

## Headline Results

- **Mortality prediction**: Surpasses supervised baselines under severe class imbalance (e.g., ~2.1% mortality), with DPT providing major gains.
- **Digital twins**: Achieves near-perfect distributional alignment (reported PCC up to 1.00) without memorization behavior.
- **Privacy**: More resistant to presence and attribute disclosure attacks than nearest-neighbor-style baselines.
- **In-silico trial rehearsal**: Recommended interventions outperform NCCN-style standard-of-care policies in synthetic rare-disease melanoma cohorts.

## Supported Clinical Trial Datasets

ClinicalTrials.gov IDs:

- `NCT00003299`
- `NCT03041311`
- `NCT00174655`
- `NCT00079274`
- `NCT00694382`

Data modalities include multivariate time-series such as adverse events, vitals, and laboratory measurements.

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Configure API keys

```bash
cp utils/config.py.example utils/config.py
# Then edit utils/config.py and add your API keys
```

### 3) Run the pipeline

```bash
# Basic run
python main.py --dataset NCT00694382 --start_idx 0 --end_idx 10

# Specify LLM backend
python main.py --dataset NCT00694382 --llm gpt-4o-mini --group experimental

# Specify task
python main.py --dataset NCT00694382 --task mortality_prediction
```

### Common CLI arguments

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset identifier | `NCT00694382` |
| `--task` | Task type | `mortality_prediction` |
| `--llm` | LLM backend name | from config |
| `--group` | Experiment group | `experimental` |
| `--start_idx` | Start patient index | `0` |
| `--end_idx` | End patient index | all |
| `--clinician_num` | Number of clinician agents | `3` |
| `--max_round` | Max collaboration rounds | `3` |

## Supported LLM Backends

- **DeepSeek**: `deepseek-chat`, `deepseek-reasoner`
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`
- **Qwen**: `qwen-plus`, `qwen-turbo`
- **Gemini**: `gemini-2.0-flash`, `gemini-2.5-flash-lite`

## Output Structure

```text
data/outputs/{dataset}/{task}/{llm}/{corpus}/{group}/
├── result.json
├── results.txt
└── {patient_id}/
    ├── clinician{1,2,3}_review.json
    ├── clinician{1,2,3}_review_messages.json
    ├── meta_summary.json
    ├── meta_final_summary.json
    ├── supervisor_logits_json.json
    └── supervisor_userprompt.txt
```

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
