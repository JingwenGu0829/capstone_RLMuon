# capstone_RLMuon

Why do spectral optimizers (Muon) struggle in RL fine-tuning of LLMs, while Adam/SGD do fine?
Working thesis: optimizers act as signal filters. The RLVR policy gradient has a low-rank signal buried in full-rank noise, and momentum cannot denoise it. Muon's orthogonalization then gives noise directions the same gain as signal directions, and the cost shows up as KL / forgetting.

- [docs/proposal.md](docs/proposal.md): thesis, theory, literature, full experiment design
- [docs/engineering_plan.md](docs/engineering_plan.md): preliminary-round engineering plan (one run per arm)
- `experiments/toy/`: CPU numpy pilots (rank-ceiling bound, low-rank-teacher bandit)
- `third_party/miles`: RL framework ([radixark/miles](https://github.com/radixark/miles)) as a git submodule. **Not modified.** Our code plugs in through miles' hook arguments.

## Layout (planned)

```
capstone/        our python package (optimizer variants, miles hooks, data prep)
scripts/         launchers for RL / SFT arms and parallel waves
measure/         Exp 1: gradient signal/noise spectrum (standalone)
eval/            checkpoint evaluation (GSM8K acc, KL to ref, forgetting)
analysis/        notebooks / figures
experiments/toy/ CPU pilots
third_party/     miles submodule
```

## Setup

```bash
git clone --recurse-submodules https://github.com/JingwenGu0829/capstone_RLMuon.git
```
