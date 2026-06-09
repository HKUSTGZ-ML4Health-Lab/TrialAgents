# TrialAgents

**Zero-Shot Forecasting of Evolving Patient Dynamics and Clinical Outcomes via Multi-Agent Collaboration**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TrialAgents simulates a multidisciplinary clinical team with four LLM-driven agents to predict outcomes, generate privacy-preserving digital twins, and rehearse trial decisions without task-specific fine-tuning.

## Core Architecture

TrialAgents follows a **4-agent, 3-stage workflow**:

## Agents:

- Supervisor: Orchestrates collaboration among Clinicians and makes final decisions.
- Clinician: Performs evidence-based and expert-level analysis.
- Trialist: Manages the dynamic deliberation report and updates the self-reflective memory bank.
- Simulator: Generates personalized and counterfactual patient trajectories.

### Stage 1 - Evidence-Based Review

Each Clinician retrieves domain evidence (via MedCPT) and produces an independent initial assessment.

### Stage 2 - Multi-Round Collaborative Deliberation

The Supervisor coordinates iterative discussions, tracks disagreements, and drives consensus formation across rounds.

### Stage 3 - Synthesis of Patient Trajectories

| Task | Component | Description |
|---|---|---|
| Mortality Prediction | DPT (LNL pipeline) | Refines noisy zero-shot outputs with lightweight temporal modeling |
| Digital Twin Generation | Simulator | Produces autoregressive, privacy-preserving patient trajectories |

## Supported Clinical Trial Datasets
ClinicalTrials.gov IDs:

- NCT00003299
- NCT03041311
- NCT00174655
- NCT00079274
- NCT00694382

Data include multivariate time-series such as adverse events, vitals, and laboratory measurements.

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
python main.py --dataset NCT00694382 --llm deepseek-chat --group experimental
```

### Common CLI arguments

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset identifier | `NCT00694382` |
| `--llm` | LLM backend name | from config |
| `--group` | Experiment group | `experimental` |
| `--start_idx` | Start patient index | `0` |
| `--end_idx` | End patient index | all |
| `--clinician_num` | Number of clinician agents | `3` |
| `--max_round` | Max collaboration rounds | `3` |


## Output Structure

```text
data/outputs/{dataset}_{task}_{llm}_{corpus}_{group}/
├── result.json
├── results.txt
└── {patient_id}/
    ├── clinician{1,2,3}_review_userprompt.txt
    ├── clinician{1,2,3}_review.json
    ├── clinician{1,2,3}_round{n}_review_userprompt.txt
    ├── clinician{1,2,3}_round{n}_review.json
    ├── supervisor_round{n}_summary_messages.json
    ├── supervisor_round{n}_summary.json
    ├── supervisor_userprompt.txt
    ├── trialist_final_summary.json
    ├── trialist_summary.json
    ├── trialist_userprompt.txt
    └── zz_final_logits.json
    
```

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
