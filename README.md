# Conformal Multi-Agent Deliberation

This repository contains the research implementation for conformal calibration
of multi-agent language model deliberation. The framework aggregates
verbalized confidence distributions across agents and debate rounds, then uses
split conformal prediction to construct prediction sets for selective
automation or human escalation.

> This code is being released solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated publication. Pull requests are not being accepted in order to maintain the code exactly as it was used in the paper.

## Method

The implementation follows four stages:

1. **Multi-agent deliberation:** agents produce probability distributions over
   candidate answers and reconsider them across debate rounds.
2. **Social aggregation:** agent distributions are combined using uniform,
   entropy-based, or consistency-based weighting.
3. **Conformal calibration:** a held-out labeled split determines a
   nonconformity threshold for a target miscoverage rate.
4. **Action policy:** singleton prediction sets can be automated, while larger
   sets can be escalated for human review.

## Repository Structure

- `main.py`: deliberation, aggregation, conformal calibration, and action-policy
  components.
- `run_evaluation.py`: command-line evaluation workflows.
- `config.py`: Amazon Bedrock model aliases and shared prompt configuration.
- `utils.py`: Bedrock response handling and dataset loaders.
- `*.sh`: experiment scripts used with the evaluation workflow.

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Model inference uses the Amazon Bedrock Converse API. Configure AWS credentials
through a standard AWS SDK credential provider and enable the selected models
in the configured region. Model aliases and Bedrock model IDs are defined in
`config.py`.

## Datasets

The code contains loaders for:

- MMLU-Pro
- GPQA
- ARC
- MuSR

GPQA is gated. Accept its dataset terms on Hugging Face and authenticate with
the Hugging Face CLI before using it.

## Evaluation Modes

The evaluation code implements:

- `greedy`: single-agent top-1 evaluation;
- `self-reflection`: iterative single-agent reflection;
- `debate-majority`: multi-round deliberation with majority voting; and
- `debate-conformal`: conformal calibration over saved debate results.

Example commands from the research workflow:

```bash
python run_evaluation.py \
  --mode debate-majority \
  --agents claude-haiku deepseek-r1 qwen-32b \
  --dataset mmlu-pro \
  --category physics \
  --rounds 3 \
  --use-subset \
  --num-samples 10 \
  --save-file results/debate-majority.json

python run_evaluation.py \
  --mode debate-conformal \
  --results-file results/debate-majority.json \
  --alpha 0.1 \
  --score-type probability \
  --save-file results/debate-conformal.json
```

## Snapshot Note

This repository is a snapshot of the committed internal `main` branch used for
publication. That snapshot imports
`data.negotiation.load_negotiation.load_negotiation_data` from
`run_evaluation.py`, but the corresponding `data` package was not committed to
that branch. The research source is preserved unchanged rather than modifying
the archived implementation.

Running evaluations may incur Amazon Bedrock charges and download third-party
datasets. Generated JSON results and log files are excluded by `.gitignore`.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for information
about reporting security issues.

## License

This project is licensed under the Creative Commons
Attribution-NonCommercial 4.0 International license. See [LICENSE](LICENSE).
