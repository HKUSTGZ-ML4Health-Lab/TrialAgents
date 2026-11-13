# TrialAgents: Agentic forecasting of patient dynamics in clinical trials with digital twins

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TrialAgents is an evidence-grounded multi-agent framework for clinical endpoint forecasting and treatment-conditioned patient trajectory simulation.
Four LLM-driven agent roles operate through three stages. Medical evidence retrieval (MERT) and the diagnostic predictive tool (DPT) are optional tools, not extra agents.

## Core Architecture

TrialAgents follows a **three-stage workflow**:

### Agent roles

- Clinician (×3): Assesses the case from oncology, emergency medicine, and comprehensive care. Each assessment includes an event probability, a confidence score, and a clinical rationale.
- Trialist: Synthesizes clinician assessments into the deliberation report and dynamically updates it each round.
- Supervisor: Orchestrates multi-agent collaboration, decides whether to continue or stop deliberation, emits the final report, and can invoke DPT for endpoint forecasting.
- Simulator: Generates individualized trajectories conditioned on observed patient information, a specified treatment, and the final report.

### Stage 1 - Medical evidence-based assessments

The three Clinician agents produce independent initial assessments. The Trialist integrates them into an initial report. The main experiment does not retrieve external evidence; `--mert` turns MERT on.

### Stage 2 - Multi-round collaborative deliberation

Each round, Clinicians review the current report and update their assessments; the Trialist refines the report; the Supervisor stops when consensus is reached or `--max_round` is exhausted. The final report contains the event probability, confidence score, and structured rationale.

### Stage 3 - Forecasting and simulation of patient trajectories

| Task | Configuration | Description |
|---|---|---|
| Endpoint forecasting | Agent-only | Uses the event probability in the final report |
| Endpoint forecasting | DPT-assisted | Supervisor invokes DPT and adopts its estimate |
| Trajectory simulation | Simulator | Treatment-conditioned longitudinal trajectories from observed patient information and the final report |

## Supported Clinical Trial Datasets 
with clearly specified prediction time windows across multiple clinical endpoints.

- NCT03041311: OS, PFS
- NCT00981058: OS, PFS
- NCT00174655: OS, DFS
- NCT00694382: OS, VTE 
- NCT00312208: OS, DFS
- NCT00003299: OS, PFS
- NCT00079274: OS, DFS

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
python main.py --dataset NCT00694382 --endpoint OS --landmark Q0_baseline --start_idx 0 --end_idx 10

# Specify LLM backend
python main.py --dataset NCT00694382 --llm deepseek-reasoner --arm treatment

```

### Common CLI arguments

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset identifier | `NCT03041311` |
| `--endpoint` | Endpoint (`OS`, `PFS`, `DFS`, `VTE`) | `OS` |
| `--landmark` | Landmark (`Q0_baseline`, `Q1_early`, `Q2_mid`, `Q3_late`) | `Q1_early` |
| `--llm` | LLM backend name | from config |
| `--arm` | Hypothetical trial arm (`treatment` / `control`; `--group` and `experimental` are aliases) | `control` |
| `--start_idx` | Start patient index | `0` |
| `--end_idx` | End patient index | all |
| `--clinician_num` | Number of clinician agents | `3` |
| `--max_round` | Max collaboration rounds | `3` |
| `--mert` | Enable MERT L0–L2 retrieval (trial card + MSD + optional CT.gov/PubMed) | off |
| `--memory` | Inject Memory Bank lessons into Phase 1/3 prompts (writes to `*_memory` dir) | off |


## Output Structure

```text
data/outputs/responses/response/{NCT}_v2/{endpoint}/{landmark}/{llm}_{arm}/
├── result.json
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

## Running time

Mean per-patient inference time for overall survival (OS) forecasting, averaged across trials and longitudinal landmarks (T0–T3).

| LLM backbone | Mean inference time per patient (s) |
|---|---:|
| DeepSeek-V4-Flash | 19.31 |
| Claude-Sonnet-5 | 12.89 |
| GPT-5.5 | 25.18 |
| DeepSeek-V4-Pro | 58.45 |
| Claude-Opus-5 | 12.66 |

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
