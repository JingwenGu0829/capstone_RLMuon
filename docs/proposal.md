# Capstone proposal: Why Muon struggles in RL — optimizers as signal filters

*Draft 2026-10-31. Rewritten from the Dec 2025 shelved draft "Unified framework" (entropy-gated RLVR updates).*

---

## 0. One-sentence thesis

> **An optimizer is a filter that decides which directions of a noisy gradient become movement.** Muon's orthogonalization (msign) equalizes every singular direction, so it is only safe when its input has already been denoised by momentum. In RLVR, the signal is low-rank (≈1 bit/episode, carried by a few forking tokens), the noise is full-rank, and gradients at adjacent steps are nearly uncorrelated (so momentum can't denoise). msign therefore gives the noise directions the same gain as the signal directions. The cost of that is paid in KL, clipping, and forgetting, not necessarily in the final reward curve.

The capstone doesn't try to "discover" this mechanism. Pion (2605.19282) already proposed it as a hypothesis. What we do is: **(1) give it a precise, falsifiable theoretical form; (2) measure directly, for the first time, the signal/noise spectral structure of RLVR gradients (no one has done this yet); (3) use a controlled experiment to separate the four competing explanations.**

---

## 1. Status of the original draft (honest assessment)

| Original content | Status as of 2026-10 | What to do |
|---|---|---|
| §3 entropy gates update magnitude, E‖∇_z log π‖² = 1−‖π‖² | **Already published**: Shen 2509.03493 (Sep 2025, before your draft) gives the 2H(π) / 2(1−‖π‖²) bound; Yang et al. 2505.12929 gives the (1−π_y) bound; GMTS 2608.30632 uses it explicitly to explain the 80/20 result | Demote to a background lemma with citations; it is no longer a contribution |
| §3 entropy-weighted KL concentration > 0.7 | Challenged by Meng et al. 2603.22446 (signed Δlog p identifies critical tokens better than entropy/KL; high-divergence contexts span all entropy levels) | Optional; if kept, add a control |
| §4 cross-seed directional coherence, split by entropy band | **No equivalent found**: still novel | Keep and upgrade (§4 E1 below), and add norm-matched and random-20% token controls (cf. Li et al. 2605.07660) |
| §5 unified view (80/20 ↔ Path Not Taken ↔ Activation Replay) | The area is now crowded (Path Not Taken, PULSE, Sparsity Curse, AlphaRL, …) | Rewrite as related work; don't pitch it as a unified framework |
| The abandoned "Adam has a problem in RL" angle | Your finding ("Adam is fine") is consistent with the literature: SGD ≥ AdamW in RLVR (Mukherjee 2602.07729) | **This turns into a positive result**: Adam/SGD are fine because they don't equalize the spectrum (see §3) |

---

## 2. Literature landscape: is Muon bad in RL? (not settled)

**Negative evidence**
- **Pion** — Fan et al., 2605.19282, May 2026. Muon on Qwen3-1.7B/4B with GRPO on GSM8K/MATH: accuracy "near zero". They attribute this to low-SNR gradients plus per-head specialization; the fix is high-pass Newton-Schulz (keep top singular values at 1, suppress the tail).
  - Weaknesses: all optimizers share one LR (1e-6) with no Muon LR sweep; the update-scale convention isn't stated; runs are only 40–80 steps; the base model is Adam-pretrained; **they never measure the RLVR gradient spectrum**.
- Jen Wei's HF blog (Feb 2026, anecdotal): Muon on Qwen2.5-0.5B with DAPO shows high entropy, format collapse, and gradient-norm spikes. Her hypothesis is "no variance state".

**Positive / neutral evidence**
- **Ruan et al.** 2607.16169 (Jul 2026), ALFWorld with Qwen2.5 0.5–3B:
  - Muon at lr 3e-5 beats AdamW at 1e-6, but **the gain disappears once update RMS is matched**.
  - The gain is larger with fine-grained credit assignment (GiGPO).
  - Normalized stable rank of the update: AdamW ≈ 0.015, Muon ≈ 0.6.
- Kimi K2, INTELLECT-3 and ZAYA1 all use Muon for RL, **but their bases are Muon-pretrained**, and none has an ablation. ZAYA1 explicitly sets RL momentum to 0.
- PufferLib 3.0 (deep RL from scratch): Muon is better after tuning.

**Confounds (must be controlled)**
1. **Update scale / LR convention**: Su Jianlin's guide (spaces.ac.cn/archives/11416) notes that with non-Moonlight scaling, LR has to be about 10× larger, and many "Muon doesn't work" reports come from this. Ruan et al. confirm it in RL.
2. **Optimizer mismatch**: Moonlight 2502.16982, Qu et al. 2605.10468 and Liu et al. 2605.06654 all show that an Adam-pretrained model → Muon fine-tune is already worse **in SFT**. Qu et al. also find that **LoRA closes the gap**. Note: the original draft's experiments used LoRA, so they can't be reused directly.
3. **Momentum fails in RL**: Mukherjee 2602.07729 measures the cosine between momentum and the current gradient at ≈ −0.007 in RL vs 0.997 in SFT. MiniMax-M1 reports weak correlation between adjacent-step gradients.
4. **Precision**: PULSE 2602.03839 shows that at RL learning rates about 99% of Adam's per-step updates are absorbed by the bf16 cast.

**Structure of the RL signal (supports "signal is low-rank")**
- LoRA Without Regret (Thinking Machines, 2025-09): rank-1 LoRA is enough for RL, at about 1 bit per episode.
- AlphaRL 2510.00553: the rank-1 component recovers ~99% of the gain.
- RELEX 2605.21468: rank-1 extrapolation denoises.
- Linear Dynamics 2601.04537.
- **But**: Mukherjee 2505.11711 finds RL updates are sparse but **nearly full-rank**.
  - Our explanation: the full rank comes from noise. Adam/SGD's accumulated updates contain the noise; the *signal* subspace is low-rank.
  - This exact distinction is what E1 measures.

---

## 3. Theory (the part that is ours)

Notation: one step's gradient estimate is Ĝ = S + N (S = true gradient, rank k; N = noise, typically full rank r = min(m, n)). The optimizer outputs Φ(Ĝ).

### Lemma 1 (spectral equalization has a rank ceiling)

By Hölder duality (spectral norm ↔ nuclear norm), using ‖msign(Ĝ)‖₂ = 1 and ‖msign(Ĝ)‖_F = √rank(Ĝ):

$$\cos\big(\mathrm{msign}(S+N),\,S\big)\;\le\;\frac{\|S\|_*}{\|S\|_F\sqrt{\mathrm{rank}(S+N)}}\;\le\;\sqrt{k/r}.$$

- The bound **does not depend on SNR**: no matter how strong the signal is, once noise makes the matrix full-rank, Muon's per-step alignment with the signal is capped at √(k/r).
- Elementwise sign (≈ Adam) has no such ceiling. Its mean-field response is E[sign(g)] ≈ g/√(g²+σ²) (Su, /archives/11280), which depends only on per-coordinate SNR and is **rank-agnostic**.
- SGD's alignment → 1 as SNR increases.
- Numerically checked (`experiments/toy/check_holder_bound.py`, 2000 random cases, 0 violations). Static toy (`experiments/toy/toy_rank_ceiling.py`): with k=1, r=256 and Frobenius SNR = 3, cos is SGD 0.95, sign 0.59, **Muon 0.06 ≈ √(1/256)**.

### Link to Su Jianlin's Muon mean field (/archives/11285, 11301)

- In expectation, E[msign(Ĝ)] ≈ G(GᵀG + (nσ²/B)I)^{-1/2}. Each singular direction is soft-thresholded by its own SNR, so **in expectation Muon is SNR-aware**.
- But every realization satisfies ΦᵀΦ = I, so the update energy is always r. The signal's share is only Σ_k s_k²/(s_k²+nσ²/B) / r.
- Su's conclusion "Muon behaves like SignSGD under noise" assumes isotropic per-entry noise and Muon's own Hessian, under which noise directions are free.
- **Low SNR alone therefore doesn't separate Muon from Adam** (pretraining is also low-SNR, and Muon works there). **What separates them is signal rank, together with the price of moving in noise directions.** This is our key correction to the Pion story.

### Lemma 2 (where the price of noise directions comes from)

Under a second-order model with curvature H, the best achievable one-step improvement is ⟨S,Φ⟩² / (2⟨Φ, HΦ⟩).

- Under Muon's implicit spectral geometry, noise directions cost nothing.
- In RL, however, a step is priced in **KL / Fisher** (the PPO clip and the KL penalty measure exactly ½ΔθᵀFΔθ; this is eq. (3) of your original draft).
- Path Not Taken shows that the pretrained model has high curvature along principal directions, and full-rank noise inevitably hits them.
- Prediction: Muon's waste shows up as **KL / clip fraction / forgetting per unit of reward gain**, not necessarily in final reward.

### Lemma 3 (momentum is the precondition for msign)

- "Denoise First, Orthogonalize Later" (2606.03899) argues that momentum is the spectral filter that lets Muon work in pretraining.
- Su's EMA post shows that momentum ≈ a batch size larger by a factor of (1+β)/(1−β), **provided adjacent gradients share the signal**.
- In RL, the momentum–gradient cosine is ≈ 0 (on-policy nonstationarity + 1-bit rewards), so this filter fails, and msign takes in a matrix that is noise-dominated and full-rank.

### The unified picture: three kinds of filtering

| Filter | SGD | Adam | Muon | Shampoo/SOAP |
|---|---|---|---|---|
| Temporal (momentum) | yes / optional | yes | yes (**its denoising relies on this**) | yes |
| Coordinate-wise SNR (second moment) | no | **yes**: β1=β2 gives "SNR-aware steepest descent" (Su /archives/11593) | no | accumulated |
| Spectral equalization | no (preserves the spectrum → naturally low-rank updates) | mild (99% energy in ~85 singular directions vs ~25 for SGD; Mukherjee 2602.07729) | **full** | partial, with accumulated statistics |

RL breaks the temporal filter and the signal is low-rank, so we expect an ordering of the form **SGD ≳ Adam > Shampoo/SOAP > Muon** in RL, and much smaller gaps in SFT. This matches the existing literature: SGD ≥ AdamW in RLVR, and Kron/PSGD help in deep RL (2506.15544).

### The token-level bridge (this is where the original draft fits)

Sequence-level advantages are also applied to irrelevant low-entropy tokens. Each such token's contribution is a rank-1 term with random sign and magnitude ∝ √(1−‖π‖²)·‖h‖. Together these terms make up the full-rank noise bulk. Meanwhile the signal comes from the few forking tokens, and **the number of forking tokens per batch bounds the signal rank**. Your cross-seed coherence measurement is a natural tool for telling the signal subspace apart from the noise subspace.

---

## 4. Pilot results (numpy toy, CPU, already run)

Setup (`experiments/toy/toy_rank_teacher.py`):
- The pretrained "LM head" is W0; the teacher is W0 plus a rank-k change.
- RL objective: GRPO with binary reward. SFT objective: CE on teacher labels.
- For each optimizer we take the best LR on a 5-point grid, 2 seeds.
- Drift is measured as KL(π0‖π) on contexts the teacher did not change, i.e. pure damage.

| Setting | Adam acc / drift | Muon (NS5) acc / drift | Muon (exact polar) | Muon (top-8 only) |
|---|---|---|---|---|
| RL, k=1 | 0.33 / 1.85 | 0.32 / **3.14** | 0.31 / **3.56** | 0.30 / 1.50 |
| RL, k=64 | 0.21 / 1.77 | 0.21 / **3.00** | 0.21 / **3.55** | 0.19 / 1.64 |
| SFT, k=1 | 0.34 / 1.16 | 0.34 / 1.27 | 0.34 / 1.39 | 0.35 / 1.12 |

What these numbers say:
- **Final reward is nearly identical**: the rank ceiling alone doesn't make Muon "fail".
- **In RL, Muon's off-task drift is ~1.7× Adam's; in SFT it is only ~1.1×.** This is consistent with Lemma 2: the cost goes into KL/forgetting.
- **Exact msign drifts more than NS5.** Prediction P4 holds directionally: the more accurately small singular values are pushed to 1, the worse things get.
- **Keeping only the top-8 directions (Pion-like) cuts drift back to Adam's level.**
- Caveats:
  - This is a toy with only 2 seeds.
  - The SGD LR grid was badly chosen, so SGD isn't comparable.
  - The difference is mainly in how fast drift grows with LR. At low LR the Adam and Muon frontiers are close.
  - **This has to be reproduced on a real model; don't write it up as a conclusion.**

We also learned something negative: a toy in which the true gradient is not low-rank (the first version, where W0 = W* + full-rank noise) can't test the hypothesis at all. **So the real-model experiments have to measure signal rank first (E1). Otherwise, whatever the outcome, we can't explain it.**

---

## 5. Experiment design

**Shared setup**
- Model: Qwen2.5-0.5B/1.5B-Instruct or Qwen3-0.6B/1.7B, **full fine-tuning** (not LoRA).
- RL: GRPO on GSM8K (with an optional MATH subset).
- SFT control: same model, same domain (e.g. GSM8K gold solutions).
- Muon is applied only to the 2-D hidden matrices; embeddings, the LM head and norms use AdamW (standard practice, Su /archives/11416).
- Muon uses Moonlight scaling, 0.2·√max(m,n).

### E1 — Measurement: signal/noise spectral structure of RL vs SFT gradients (core novelty, needs few GPU-hours)

At several checkpoints, for each layer type (q/k/v/o/up/down):
1. **Split-batch spectral SNR profile**:
   - Compute two independent half-batch gradients G₁ and G₂, and take the SVD of G₁ → (σᵢ, uᵢ, vᵢ).
   - ρᵢ = uᵢᵀG₂vᵢ / σᵢ is the fraction of direction i that replicates.
   - k_eff = Σᵢ ρᵢ, or the number of directions with ρᵢ > 0.5. This is the signal rank.
2. **Spectrally resolved cross-seed coherence**: an upgrade of the original Table 1. Compute cross-seed cos of ΔW separately within the top-k and the tail singular subspaces. Prediction: the top directions are coherent and the tail is noise.
3. The momentum–gradient cosine, and Su's SNR estimator ‖m‖²/‖g‖² (/archives/11267).
4. **Muon's wasted fraction** 1 − Σρᵢ/r, compared with the corresponding quantity for SGD/Adam.
5. Entropy split: decompose the gradient by token entropy band and measure each band's energy in the signal vs noise subspace (link to the original §4).

Prediction: RL has k_eff ≪ r while SFT's k_eff is much larger; momentum cosine is ≈ 0 for RL and ≈ 1 for SFT.

### E2 — Controlled optimizer comparison (main experiment)

**Arms**:
- AdamW
- SGD(+momentum)
- Muon (β=0.95)
- Muon (β=0)
- Muon (exact polar or Polar-Express)
- Muon (top-k / Pion high-pass)
- SOAP (optional)

Each arm gets a 3–4-point LR sweep, plus **an RMS-matched control** (Muon's update RMS scaled to AdamW's measured RMS).

**Metrics**:
- reward curve
- **reward–KL(π‖π_ref) frontier**
- clip fraction
- entropy curve
- **forgetting**: perplexity on held-out general text or a small capability probe
- update stable rank
- the E1 quantities

The same arms also run SFT, which gives a (RL vs SFT) × optimizer grid.

### E3 — Discriminating predictions (core of the argument)

| Explanation | RMS-matched | Same arms on SFT | NS5 → exact msign | top-k / high-pass | β: 0.95 → 0 | Larger G / finer credit |
|---|---|---|---|---|---|---|
| H1 update scale (Ruan) | gap disappears | — | no effect | no effect | — | — |
| H2 optimizer mismatch (Moonlight) | gap remains | **SFT equally bad** | no effect | small | — | no effect |
| **H3 rank ceiling + noise price (ours)** | gap remains (in KL/forgetting) | SFT much better | **worse** | **fixes it** | **little effect in RL**, large in SFT | gap shrinks |
| H4 per-head specialization (Pion) | — | — | — | partial | — | — |

H2 can only be ruled out cleanly with a Muon-pretrained base:
- Plan A: look for a small open Muon-pretrained model.
- Plan B: pretrain two 124M GPT-2 models (modded-nanogpt Muon vs llm.c AdamW). That takes a few GPU-hours. Then do SFT + RL on a synthetic verifiable task (multi-digit arithmetic / countdown) to get a 2×2 pretraining-optimizer × fine-tuning-optimizer grid.

### E4 (optional) — The SNR knob
- Vary group size G ∈ {4, 16} and prompt batch.
- Compare sequence-level advantages against a token/step-level credit signal, e.g. rewards from a process verifier, or advantages applied only to the top-20% entropy tokens.
- Prediction: Muon's deficit shrinks as k_eff rises. This is the controlled version of Ruan et al.'s GiGPO observation.

---

## 5.5 Minimal validation (4×H100, go/no-go)

Shared setup:
- Model and task: Qwen3-1.7B (non-thinking) + GSM8K GRPO. This matches the setup in which Pion reports "Muon collapses", so the result directly confirms or refutes theirs.
- **One run per GPU**: full fine-tuning of 1.7B fits on a single 80 GB card with vLLM colocated. No sharding is needed, so msign / SVD are trivial, and 4 GPUs = 4 runs in parallel.
- fp32 master weights for every arm (to avoid the PULSE bf16-absorption confound).
- KL coefficient β = 0, but KL is measured.
- Muon is applied only to hidden matrices.

| # | Experiment | Runs | GPU-h | Question it settles |
|---|---|---|---|---|
| 1 | **Gate measurement**: at init (and at an AdamW checkpoint around step 100), compute many pairs of independent half-batch gradients (G₁, G₂) for both RL and SFT. Measure per-layer k_eff, plus each map's alignment with the true gradient, estimated without bias by ⟨Φ(G₁), G₂⟩ / (‖Φ(G₁)‖·√⟨G₃, G₄⟩). The maps are: identity, sign, msign, top-k msign. Also split by token entropy band. | 0 training | ~6 | Is the RL signal much lower-rank than SFT's? Is msign's alignment loss RL-specific? **If this fails, stop.** |
| 2 | **RL sweep**: 4 arms (AdamW / Muon-NS5 / Muon-exact / Muon-top-k, with k from #1) × 3 LRs, 200 steps; then the best LR for each arm with +1 seed. Every step logs the momentum–gradient cosine. | 16 | ~30–50 | Is the reward–KL / reward–forgetting frontier dominated by Muon across the whole LR sweep (this rules out the scale confound)? Is exact ≥ NS5 in cost? Does top-k close the gap? |
| 3 | **SFT control**: the same 4 arms × 2 LRs on GSM8K gold solutions | 8 | ~5 | Is the Muon−AdamW gap specific to RL (separating H3 from H2)? |

Wall clock: about 1–1.5 days. Engineering (optimizer variants + measurement script + the TRL pipeline) takes about a week and is the real cost.

Deliberately cut: SGD (literature already shows SGD ≥ AdamW), SOAP, Muon β=0, the G sweep, and a Muon-pretrained base.

Interpreting the outcomes:
- #1 fails → the rank story doesn't hold. Pivot to a "confound audit" paper (Muon vs AdamW in RLVR with LR/RMS controlled).
- #1 passes but the #2 frontiers overlap → the mechanism exists but is harmless at this scale. Contribution: the measurement, plus a refutation of "Muon collapses".
- Muon is equally bad in #2 and #3 → mismatch (H2) dominates.
- Muon is worse only in RL, exact is worse, and top-k fixes it → the thesis is supported; expand to the full plan.

---

## 6. Compute and timeline (rough)

- **Tier 0 (CPU, done/extendable)**: the toys. Extend to a 2-D phase diagram of k × noise × "price" (KL penalty on/off).
- **Tier 1 (core, about 60–100 H100-hours)**:
  - 0.5B model, GRPO, ~150–200 steps per run, about 1.5–2.5 h per run (vLLM rollouts).
  - About 30 runs in total: 7 arms × 3 LRs, plus 2 extra seeds for the best configuration.
  - SFT controls and E1 measurements are cheap.
  - On Modal at list H100 price this is roughly a few hundred dollars. **Check current pricing yourself.**
- **Tier 2 (optional)**: 124M pretraining 2×2 (H2); 1.5B replication.
- **Timeline (~10 weeks)**:
  - W1–2: theory write-up + toy phase diagram + get the RL/SFT pipeline running with Muon.
  - W3–4: E1 measurements.
  - W5–7: E2 sweeps.
  - W8: E3 ablations.
  - W9–10: writing.

**Fallback**: if Muon doesn't lose to Adam in our setting (possible, given Ruan et al.), the paper becomes "the reported Muon failures in RLVR are explained by X (scale / mismatch); the real cost is in KL/forgetting". That is still a complete result.

---

## 7. New paper outline

1. Introduction: Muon works in pretraining, but RL reports conflict. There are four competing explanations.
2. Background:
   - entropy-gated token gradients (cite Shen / Yang / GMTS; your old §3 condensed into one lemma)
   - the three kinds of optimizer filtering
3. Theory: Lemma 1 (rank ceiling), its relation to Su's mean field, Lemma 2 (noise price in KL geometry), Lemma 3 (momentum precondition).
4. Measurement: the signal/noise spectral structure of RLVR vs SFT gradients (E1). Your cross-seed coherence lives here.
5. Controlled experiments and discriminating ablations (E2/E3).
6. Discussion: when Muon is usable in RL (Muon-pretrained bases, token-level credit, filtered variants), and why SGD/Adam are fine.

---

## 8. Reading list

Verified directly (abstract pages read): Pion 2605.19282; Ruan et al. 2607.16169; Mukherjee "Do We Need Adam?" 2602.07729. Details beyond the abstracts (Pion's single LR / 40–80 steps, Ruan's stable-rank numbers, Mukherjee's momentum cosine and singular-direction counts) and every other entry come from reading by the literature-review agents; **open each one before citing it**.

**Muon / optimizers in RL**
- Moonlight 2502.16982
- Kimi K2 2507.20534
- INTELLECT-3 2512.16144
- ZAYA1 2605.05365
- Qu et al. 2605.10468
- Liu et al. 2605.06654
- Kron/PSGD in deep RL 2506.15544
- Adam on Local Time 2412.17113
- MiniMax-M1 2506.13585

**Muon theory**
- HTMuon 2603.10067
- Denoise First, Orthogonalize Later 2606.03899
- Muon tail-end associative memory 2509.26030
- Kunstner et al. 2304.13960
- Balles & Hennig 1705.07774

**RL update structure**
- Path Not Taken 2511.08567
- RL Subnetworks 2505.11711
- PULSE 2602.03839
- RL's Razor 2509.04259
- AlphaRL 2510.00553
- RELEX 2605.21468
- Linear Dynamics 2601.04537
- Sparsity Curse 2606.18521
- LoRA Without Regret (thinkingmachines.ai/blog/lora)

**Entropy and tokens**
- Shen 2509.03493
- Yang et al. 2505.12929
- GMTS 2608.30632
- RSI 2606.31575
- Entropy Mechanism 2505.22617
- 80/20 2506.01939
- Meng et al. 2603.22446
- Li et al. 2605.07660

**Su Jianlin (spaces.ac.cn/archives/…)**
- 10592: Muon appreciation; msign; Muon = Shampoo without accumulation
- 10739: Muon sequel; RMS matching; fine-tuning mismatch
- 11285: LR & batch size (3): Muon
- 11301: LR & batch size (4): EMA
- 11267: Adam update RMS = 0.2, plus the SNR estimator
- 11593: β1=β2, SNR-aware steepest descent
- 11416: Muon guide; LR scaling; LM head should not use Muon
- 11126: QK-Clip; full-rank updates and the mismatch explanation
- 11917 / 11964: "Slow is fast"; a counter-argument to consider
