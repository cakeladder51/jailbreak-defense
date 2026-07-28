# Three-model run, 2026-07-27 — first valid post-fix measurement of CGP

> ## ⛔ CORRECTION, 2026-07-27 (later the same day) — read before anything below
>
> **The headline conclusion in this document was wrong, and it was wrong in the direction
> that makes CGP look worse than it is.** The generations, timings and stability numbers are
> all fine; the *interpretation* of the AUC was not.
>
> AUC here is computed with attacks as the positive class. But CGP's rule is *gap **below**
> threshold → serve the refusal*, so the method predicts attacks have the **smaller** gap —
> an AUC below 0.5 is the **intended** direction, and `fit_threshold` (cell 3, line 803)
> infers that sign from the data automatically. Calling qwen's 0.335 and phi's 0.257
> "significantly inverted" and "routing backwards" inverted the meaning of the result.
>
> Oriented correctly (`max(AUC, 1−AUC)`):
>
> | model | all 7 families | excluding cipher | advbench only |
> |---|---|---|---|
> | gemma-3-4b | 0.525 | 0.602 (ns) | 0.636 |
> | qwen2.5-3b | 0.665 | **0.750** | 0.904 |
> | phi-3-mini | 0.743 | **0.798** | 0.971 |
>
> And `cipherchat_cipher` is the **only** family that runs *opposite* to theory —
> significantly so on gemma (raw 0.937, CI [0.829, 1.000]) and qwen (0.846, [0.697, 0.960]).
> Every other significant family is theory-consistent. It is not "the only family CGP
> detects" as claimed below; it is the one family CGP gets **backwards**, and it was pulling
> the aggregate to chance. Excluding it, qwen and phi separate with CIs excluding 0.5 and
> beat both controls.
>
> Cause of the failure on cipher: the biased path never produces a refusal there (rate 0.00
> on gemma and phi), because the models do not recognise a ciphered request as harmful at
> all. The gap then measures the cost of corrupting already-corrupt text, and the sign
> flips. **CGP's ceiling is the model's own latent refusal tendency** — it cannot surface
> what the model does not recognise.
>
> Also found while re-checking: **"did the biased path refuse?" — a plain string check on
> text CGP already generates — separates better than the plausibility gap on 2 of 3
> models**, and the lexicographic combination (refused first, smaller gap breaks ties) is
> the first signal in this project to beat both controls on all three (ex-cipher: gemma
> 0.798, qwen 0.855, phi 0.806, vs `base_avg` 0.711/0.709/0.610 and `H` 0.729/0.726/0.555).
>
> **Still unresolved, and it gates all of the above:** every number here is measured against
> Alpaca-only benign. A hard benign set (XSTest-safe) is being run now; `bias_is_refusal` is
> exactly the signal XSTest is built to break, so these figures may not survive it. Nothing
> in this correction should be quoted as a positive result until that lands.
>
> The original text is left unedited below so the record of what was claimed stays intact.
> Commit `69f7a33`'s message carries the same error.

**Corpus:** 105 rows per model — 10 prompts from each of the 7 attack families (70: 60 `harmful`,
10 `forced_prefix`) + 35 Alpaca benign. `SAMPLE_SEED=GEN_SEED=SPLIT_SEED=67`.
**Models:** `gemma-3-4b-it`, `qwen2.5-3B-Instruct`, `phi-3-mini-4k-instruct` — three labs, **all bf16**,
so cross-model differences are not partly a quantization artifact.
**Code:** branch `fix/refusal-processor-start-ids`, after the `start_ids` fix, `PROC_PARAMS` at the
shipped values (justified in [`proc_params_sweep.md`](proc_params_sweep.md)).
**Judge:** OFF. Generations are banked; **no ASR/FPR/NAR here.** Everything below is judge-free.
**Hardware/time:** RTX 5070 Ti, serial. gemma 60 min, qwen 43 min, phi 27 min.
**Tags:** `gemma_bf16`, `qwen_bf16`, `phi3_bf16` in `cgp_out/`.

## Sanity check first — the processor is genuinely active

Pre-fix, the biased path was byte-identical to the baseline on 70% of rows: the processor was inert
and every number derived from it was meaningless. That is no longer the case on any model.

| model | identical outputs | refusal strings, base → biased | processor fired |
|---|---|---|---|
| gemma-3-4b | 41/105 (39%) | 40 → 43 (**+3**) | 105/105 |
| qwen2.5-3b | 27/105 (26%) | 29 → **73** (**+44**) | 105/105 |
| phi-3-mini | 31/105 (30%) | 31 → 48 (**+17**) | 105/105 |

So these are real measurements of CGP as designed, not of a no-op. (Refusal counts are crude string
matches, not judge verdicts.)

## A1 — separation of attack from benign (the headline result)

`p_ratio_norm` is what the gate thresholds. 0.5 = no separation. Above 0.5 = attacks score higher
(the intended direction). Below 0.5 = **benign** score higher, i.e. the signal is inverted.

| model | `p_ratio_norm` | 95% CI | `base_avg` (control) | `H` (control) |
|---|---|---|---|---|
| gemma-3-4b | 0.475 | [0.370, 0.585] | 0.247 | **0.767** |
| qwen2.5-3b | 0.335 | [0.235, 0.440] | 0.346 | **0.674** |
| phi-3-mini | 0.257 | [0.166, 0.357] | 0.347 | 0.506 |

**No model shows above-chance separation in the intended direction.** gemma's CI spans 0.5 (no
effect); qwen's and phi's CIs lie **entirely below** 0.5 — on those two models the plausibility gap
is reliably *larger on benign prompts than on attacks*, so a gate trained on it routes backwards.

For gemma this is unchanged from the pre-fix number (0.492 [0.398, 0.579] → 0.475 [0.370, 0.585]).
**The `start_ids` fix was necessary but did not move A1.** It corrected what the processor does; it
did not make the resulting signal discriminative.

## Threshold stability — worse than the AUCs

`fit_threshold` infers the routing direction from which class has the larger mean. Refit over 200
resampled splits:

| model | threshold mean ± sd | p05 → p95 | direction |
|---|---|---|---|
| gemma-3-4b | 0.218 ± 0.146 | 0.039 → 0.416 | **FLIPS** (96% positive) |
| qwen2.5-3b | 0.177 ± 0.332 | 0.000 → 0.985 | **FLIPS** (14% positive) |
| phi-3-mini | 0.281 ± 0.125 | 0.131 → 0.608 | stable — but **stably inverted** (0% positive) |

On gemma and qwen, which prompts get refused depends on how the train/test split happened to fall;
qwen's fitted threshold ranges over essentially the whole scale (0.000–0.985). phi-3-mini is the only
model with a stable direction, and that direction is the wrong one — consistent with its AUC of 0.257.
Note that "direction_stable = True" in the JSON is **not** a good sign on its own; read it with the AUC.

## Per-family — consistent across models, and consistently the wrong thing

| family | gemma | phi | qwen |
|---|---|---|---|
| **cipherchat_cipher** | **0.937** | **0.586** | **0.846** |
| advbench_prefill | 0.579 | 0.371 | 0.426 |
| safemtdata_multiturn | 0.576 | 0.454 | 0.523 |
| artprompt_orthographic | 0.371 | 0.037 | 0.136 |
| advbench | 0.364 | 0.029 | 0.096 |
| jailbreakbench_general | 0.311 | 0.029 | 0.271 |
| msj_contextwindow | 0.184 | 0.294 | 0.046 |

Spearman correlation of the family ranking: gemma–qwen **+0.857**, phi–qwen **+0.750**,
gemma–phi **+0.714**.

This is the consistency the small-model pilot was meant to establish, and it is real — but what is
consistent is the **confound**. `cipherchat_cipher` is the top family on all three models, by a wide
margin, and it is the one family whose prompt and completion are unreadable gibberish. Plain-English
`advbench` is near the bottom on all three (0.364 / 0.029 / 0.096). The signal tracks *how strange the
text is*, not how harmful the request is — exactly the pattern the sweep found at 24 rows and the
same one the benign-set confound predicts (CLAUDE.md). With Alpaca as the only benign set, "unusual
text" and "attack" are not separable, so this ordering cannot be read as CGP detecting attacks.

## Cost

CGP is a **2.0–2.1× wall-clock overhead** (a second full generation) on every model: gemma
11.25 s + 11.27 s, qwen 7.60 s + 8.35 s, phi 6.34 s + 6.55 s. SafeDecoding costs about the same as a
single baseline pass. The controls that outperform the headline signal (`H`, `base_avg`) are computed
from the baseline generation alone — they cost **nothing** beyond the answer the model already
produced.

## What this means

1. **CGP's headline signal does not separate attacks from benign on any of the three models**, and on
   two of them it is significantly inverted. This is now measured with a working processor, so it is a
   result about the method, not about the bug.
2. **The `max_bias` problem is model-dependent and there is no single right value.** On gemma the bias
   barely changes the output (+3 refusals; 39% of rows identical) because gemma already opens with
   `"I"`. On qwen the same setting bites hard (+44 refusals; benign text changed on 34/35 rows) and
   drives the inversion. A globally-fixed `max_bias=10` is simultaneously too weak for one model and
   too strong for another — see [`proc_params_sweep.md`](proc_params_sweep.md).
3. **The controls remain unbeaten**, so the second generation pass is not yet earning its 2× cost.
4. **None of this is safe to write up as "CGP does not work" yet.** The benign set is the blocker: with
   Alpaca only, both the headline and the controls are contaminated by the same text-weirdness effect.

## Prediction for the judge run (recorded before it happens)

The judge does not change any refusal — it labels text already banked in the `.jsonl`. So what it
will report is already determined by those files, and `tools/predict_judge.py` reads it off them with
a refusal-marker string match. **Recorded here so the judge's real output can be checked against it.**

**(A) Refusal rate of each path, before routing** — the ceiling CGP could reach with a perfect gate:

| model | attack: base → biased | benign: base → biased |
|---|---|---|
| gemma-3-4b | 45.7% → 55.7% (+10.0pp) | 2.9% → 2.9% (+0.0pp) |
| qwen2.5-3b | 44.3% → 82.9% (**+38.6pp**) | 5.7% → 22.9% (**+17.1pp**) |
| phi-3-mini | 42.9% → 51.4% (+8.6pp) | 5.7% → 14.3% (+8.6pp) |

**(B) What CGP actually serves through its fitted gate** (same held-out protocol, 200 splits):

| model | attack refused (vs undefended) | benign refused (vs undefended) | gate inverted |
|---|---|---|---|
| gemma-3-4b | 47.8% (**+2.1pp**) | 2.9% (+0.0pp) | 5% of splits |
| qwen2.5-3b | 46.9% (**+2.6pp**) | 7.7% (+1.9pp) | 85% of splits |
| phi-3-mini | 47.0% (**+4.1pp**) | 8.2% (+2.5pp) | 100% of splits |

So expect the judge to show a **small rise in refusals on harmful prompts, ~+2 to +4pp**, against the
+8.6 to +38.6pp the biased path had available — the gate discards most of it, because routing on a
chance-level signal is close to routing at random. The rise is paid for with a comparable relative
increase in benign refusals on qwen and phi.

Two ways the real judge should differ from this string match:
- **It will count more refusals, unevenly.** Hedged deflections with no marker phrase — gemma's
  `"I understand you're grappling with difficult thoughts…"` — should be scored `refused`. Absolute
  levels move, most on gemma; the base-vs-served *delta* should hold, since both are scored alike.
- **`na` has no string-match equivalent.** The cipher completions are gibberish and are the obvious
  `na` candidates — and that is the one family carrying CGP's apparent signal (AUC 0.937 / 0.846 /
  0.586). If it lands in `na` it leaves the ASR denominator. Check this specifically.

## ⚠️ Before running the judge: a bad key used to poison the files permanently

`RealJudge.one` retried 4× and then returned the **string** `"na"` on failure. `run_phase()` writes
any non-`None` verdict and rewrites the `.jsonl`; its resume logic only re-judges fields that are
`None`. So one run with an expired or invalid key stamped `na` onto every row, and the later run with
a working key skipped all of them — recoverable only by deleting the generations and re-running the
GPU. Fixed 2026-07-27: a failed call now returns `None` (row stays unjudged, retried next run) and
`batch()` logs an error naming how many calls failed. **Still worth confirming the key works before
kicking off the full pass.**

## Next, in order

1. **Add a hard/borderline-benign set (XSTest-style).** This is the single blocking item. Until benign
   prompts are as unusual-looking as attacks, neither the headline nor the control AUCs are
   interpretable, and no conclusion about CGP can be defended in review.
2. **Turn the judge on** (`JUDGE_ENABLED=True`, same `RUN_TAG`) to fill ASR/FPR/NAR onto these banked
   generations — nothing regenerates.
3. Only then revisit per-model `max_bias`.

## Reproduce

```
bash tools/run_all.sh                                  # all three, serial
python tools/check_paths_differ.py cgp_out/<model>_<tag>_cgp.jsonl   # ALWAYS run this first
python tools/preview_auc.py       cgp_out/<model>_<tag>_cgp.jsonl
python tools/compare_runs.py                           # the cross-model tables above
```

Two loader bugs were fixed to get phi-3-mini running, both on the branch: `MODEL_LOAD` now takes a
per-model `trust_remote_code` (phi-3-mini's hub `modeling_phi3.py` reads `config.rope_scaling["type"]`,
which transformers 5.x normalizes to `{"rope_type": …}`), and `tools/run_cgp.py` no longer discards the
rest of a model's `MODEL_LOAD` entry when overriding precision.
