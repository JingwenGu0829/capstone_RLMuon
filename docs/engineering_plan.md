# Engineering Plan: Preliminary Results

*2026-10-31. Goal: **run each arm once (no LR / seed sweep)** to get a rough picture of Exp 1/2/3 from the proposal. If the picture looks right, scale up.*

Principle: **never modify `third_party/miles`.** Everything goes through miles' plugin hooks plus `PYTHONPATH`. Our code lives in `capstone/`, `scripts/`, `measure/` and `eval/`.

---

## 1. Fixed setup (shared by all arms)

| Item | Setting | Notes |
|---|---|---|
| Base model | Qwen3-1.7B | Same as Pion; directly confirms or refutes their "Muon collapse" result |
| Data | `zhuzilin/gsm8k` (train/test parquet, `messages` + `label`), `--rm-type math` | Exactly what miles' own GSM8K example (`examples/infra_features/true_on_policy/run_simple.py`) uses |
| GRPO | 32 prompts × 8 samples = 256 per step; **global batch = 256 (one optimizer step per rollout, strictly on-policy)**; max response 1024; temperature 1.0; ε-clip 0.2; 200 rollout steps | One optimizer step per rollout avoids confounding from multiple off-policy minibatches |
| KL | `--use-kl-loss --kl-loss-coef 0` | KL to the reference model is logged but does not affect the gradient (**M3 must verify that coef=0 still computes it**) |
| LR | 1e-6, constant (10-step warmup); **weight decay = 0** | Removes the decoupled-WD vs. Adam-WD confound |
| Muon scale | `--muon-scale-mode spectral --muon-extra-scale-factor 0.2`, i.e. Moonlight RMS matching | Makes Adam's LR directly reusable for Muon (Su /archives/10739, 11416) |
| Muon scope | Hidden 2-D matrices only; embedding, LM head and norms use Adam | This is the default `default_param_overrides` in Megatron's registry |
| Precision | bf16 compute, **fp32 master weights** | Avoids the PULSE bf16-absorption confound (**M0 must confirm the Megatron default**) |
| Parallelism | TP=PP=CP=1, **1 GPU per run**, `--colocate` | 4×H100 = 4 runs in parallel; SVD-based variants need no TP handling |

## 2. Arms (one run each)

| Wave | GPU0 | GPU1 | GPU2 | GPU3 |
|---|---|---|---|---|
| W1 RL | **A0** adam | **A1** muon (NS5 quintic) | **A2** muon-svd (exact msign) | **A3** muon-topk (k from Exp 1, default 32) |
| W2 RL | **A4** muon, polar_express coefficients (more accurate msign without SVD) | **A5** adaptive_muon / normuon (with second moment) | **A6** muon, momentum = 0 (ZAYA1-style) | **A7** soap (accumulated preconditioner) |
| W3 SFT | **S0** adam | **S1** muon | **S2** muon-svd | **S3** muon-topk |

Expected outcomes under our hypothesis (H3):
- A1 is worse than A0 on the reward–KL / drift curves.
- A2 and A4 are no better than A1, and likely worse.
- A3 and A5 sit close to A0.
- A6 is about the same as A1, because momentum is already useless in RL.
- A7 lands somewhere in between.
- In W3, the gaps among S0–S3 are much smaller than among A0–A3.

What each arm changes:
- A2, A3, A6 and the SFT arms are each controlled by a single env var or a single CLI flag.
- A4, A5 and A7 need no new code; they are pure config changes. `--muon-coefficient-type polar_express`, `--optimizer adaptive_muon --adaptive-muon-moment2-method normuon`, `--optimizer soap`.

---

## 3. Modules

### M0. Environment (GPU, ~0.5 day)
- Use the official miles docker image. It already contains the radixark Megatron-LM fork, Emerging-Optimizers v0.3.0 and SGLang. Record the image tag in `env/IMAGE_TAG`.
- Run the official small-model smoke test (Qwen3-0.6B, `--mode debug_minimal`).
- **Pass:** one train step plus one weight sync to SGLang succeeds; the log confirms fp32 main params.

### M1. Muon variants: `capstone/optim/muon_variants.py` + `capstone/hooks/megatron_init.py`
- Megatron restricts `--optimizer` to `adam|sgd|muon|dist_muon|lion|soap|adaptive_muon`. So we **keep `--optimizer muon` and swap the orthogonalization inside it**, with no Megatron fork.
- `CapstoneMuon(TensorParallelMuon)`: after `__init__`, replace `self.scaled_orthogonalize_fn` according to the `CAPSTONE_ORTHO` env var:
  - `ns`: upstream behavior, no change.
  - `svd`: `torch.linalg.svd` in fp32, returning U Vᵀ.
  - `topk:K`: U_K V_Kᵀ.
  - (later) `highpass:K`: a Pion-style variant.
  - The scale factor stays `get_muon_scale_factor(...) × extra_scale_factor`. QKV split is handled by the parent's `orthogonalize`, so we only replace the inner function.
- Init hook (`--custom-megatron-init-path`):
  - Runs early in the actor's `init`, before the optimizer is built.
  - Replaces `megatron.core.optimizer.emerging_optimizers._EMERGING_OPTIMIZERS["muon"]` with an entry whose `optimizer_cls=CapstoneMuon`.
  - Rank 0 prints `[capstone] muon ortho=<mode>`.
- **CPU unit tests:**
  - `svd` output has all singular values equal to 1.
  - `topk` output has rank K.
  - `ns` output matches upstream exactly.
  - The scale factor is correct.
  - Registry replacement works when `get_megatron_optimizer` is called with a fake config.

### M2. Diagnostics: `capstone/hooks/diagnostics.py` (`--custom-megatron-before-train-step-hook-path`)
- The hook is called with `(args, rollout_id, step_id, model, optimizer, sched)`. On the first call it wraps `optimizer.step`. This works the same way for every arm, including Adam and SOAP.
- Watchlist: layers {1, 14, 27} × {`linear_qkv`, `linear_proj`, `linear_fc1`, `linear_fc2`}, 12 matrices in total.
- Every 5 steps, rank 0 appends to `diag.jsonl`. **Gradient side** (`main_grad`):
  - norm
  - cos(g_t, g_{t−1})
  - cos(g_t, momentum), when the optimizer state contains it
  - Su's SNR estimator (/archives/11267)
- **Update side** (ΔW = W_after − W_before):
  - RMS
  - stable rank
  - effective rank
  - share of energy in the top-{1, 8, 32} directions
  - share of energy in W₀'s top-1% principal subspace (the Path Not Taken metric; W₀'s SVD is computed on the first call)
  - stable rank of the cumulative ΔW
- Cost: 12 SVDs of at most 2048×6144 every 5 steps, which is negligible.
- **To verify:** that `main_grad` is still readable when `step` is called (Megatron DDP may have already copied it into the fp32 main param's `.grad`); handle both paths.
- **CPU unit tests:** check the metric functions on synthetic low-rank + noise matrices, reusing the `experiments/toy` data generators.

### M3. Launch: `scripts/run_rl.py`, `scripts/run_sft.py`, `scripts/launch_wave.sh`
- Written with typer, following miles' `scripts/run_qwen3_4b.py` and `run_simple.py`. Arguments: `--arm A0..A7|S0..S3`, `--mode smoke|normal`, `--gpu`, `--run-id`.
- An arm table maps each arm to its optimizer flags plus env vars, and also injects both hook paths and `PYTHONPATH`.
- Done once and shared: download the HF model and data, then convert HF → torch_dist.
- `launch_wave.sh` runs 4 arms in parallel, each with `CUDA_VISIBLE_DEVICES=i` and its own Ray port / temp dir.
  - **Risk:** 4 Ray clusters on one machine may conflict. Fallback: 4 containers with one GPU each.
- **Pass:** a 5-step smoke run for every arm. The logs contain reward, `pg_clipfrac`, `ppo_kl`, the KL-to-ref value, entropy, and the `[capstone]` optimizer line.

### M4. Data: `capstone/data/`
- RL: use `zhuzilin/gsm8k` directly.
- Confirm how to turn off Qwen3 thinking mode (`enable_thinking=False`), i.e. whether miles exposes a chat-template kwarg.
- SFT:
  - Write `prepare_gsm8k_sft.py`, which builds `messages` from openai/gsm8k's full gold solutions.
  - Read `miles/rollout/sft_rollout.py` to confirm the expected format.
  - Launch with `--loss-type sft_loss`.

### M5. Exp 1 gradient-spectrum measurement: `measure/grad_spectrum.py` (standalone, 1 GPU)
- Uses HF transformers for gradients and the SGLang offline engine for rollouts (both are in the image). It does not go through miles training.
- **RL side:**
  - 256 prompts × 8 rollouts, GRPO advantages.
  - Split into 8 disjoint groups of 32 prompts each and compute one gradient per group, recorded with hooks for the 12 watchlist matrices.
- **SFT side:** same prompts, gold solutions, same grouping.
- **Outputs:**
  - Per layer: k_eff, using split-half ρᵢ = uᵢᵀG₂vᵢ / σᵢ.
  - **Unbiased** alignment of each map (identity / sign / msign / top-k) with the true gradient, via ⟨Φ(G₁), G₂⟩.
  - The split into top-20% vs bottom-80% entropy tokens.
  - A recommended k for A3 / S3.
- **Pass:** finishes on the init model in under 2 h and writes JSON plus plots.

### M6. Eval: `eval/eval_ckpts.py`
- Convert Megatron checkpoints at steps {0, 50, 100, 150, 200} → HF using miles' conversion tools.
- Metrics:
  - GSM8K test accuracy (greedy); miles' in-training `--eval-interval` already covers this, so it doubles as a cross-check.
  - KL(π_θ‖π_ref) on 500 held-out prompts.
  - **Forgetting:** perplexity on 500 generic-text documents (wikitext-103 test or a FineWeb-edu sample).

### M7. Analysis: `analysis/prelim.ipynb`
Figures:
1. Reward vs step.
2. **Reward vs KL to ref**, the core figure.
3. Forgetting vs reward.
4. Diagnostic curves: momentum cosine, update stable rank, off-principal share.
5. Exp 1 spectra and alignment bar chart.

---

## 4. Schedule

| Time | Work | Needs GPU? |
|---|---|---|
| D1–3 | M1, M2 (with CPU tests); M3 scaffold; M4; code for M5 and M6 | No; can be done locally |
| D4 | M0 + smoke runs for every arm, plus integration debugging (expected to be most of the debugging) | 1–4 GPUs |
| D5 | Exp 1 on GPU3 while W1 A0–A2 run on GPU0–2; then A3 with k from Exp 1 | 4 |
| D5–6 | W2 (~3–5 h), W3 SFT (~1 h) | 4 |
| D7 | M6 eval + M7 analysis | 1 |

## 5. Risks and open questions

1. **Where are the GPUs?** Modal, a cluster (Slurm + enroot?), or bare metal with Docker? This determines how M0 and the 4-way parallel launch are done.
2. **A single LR is a known weakness (Pion's problem too).** Moonlight RMS matching makes 1e-6 a fairly fair common point, but the preliminary write-up must state this clearly. A sweep follows once the picture is clear.
3. Speed of the SVD arms: about 200 matrices of full SVD per step, so a few seconds per step. Acceptable; if too slow, switch to `torch.svd_lowrank` for topk.
4. Qwen3-1.7B response length and thinking mode affect rollout time.
5. Muon's QKV split defaults to True and we keep it. Per-head orthogonalization (Pion's H4) is not tested in the preliminary round.
6. Version pinning:
   - miles submodule: pinned to `c62ea5d`.
   - Megatron fork and Emerging-Optimizers: pinned by the image tag.
