# CGP — Contrastive Generative Paths for Safety

Research code for **CGP**, an inference-time defense against forced-prefix (and other) LLM jailbreaks.
For each prompt it runs two generations — a normal **baseline** path and a refusal-**biased** path —
scores each by per-token plausibility, and if the gap (the *Plausibility Ratio*) is below a fitted
threshold it serves the refusal. There is an accompanying ACL-style paper being revised to match this code.

## Key files
- **`CGP (1).ipynb`** — the whole evaluation pipeline (mock + real modes). This is the main artifact.
- **`compile_datasets.py`** — builds the adversarial benchmark → `all_prompts.jsonl` (7 attack families, 100/source).
- **`all_prompts.jsonl`** — compiled prompts consumed by the notebook. **Gitignored** (harmful content; regenerate with the script).
- **`dataset_card.docx`** — prose description of the 7 datasets / attack families.
- **`yash_aaim (1).pdf`** — the paper being updated. See "Paper" below.
- **`.env`** — API keys (`GOOGLE_API_KEY`, `HF_TOKEN`, `OPENAI_API_KEY`). **Gitignored.**

## Notebook layout (`CGP (1).ipynb`)
It's a small number of large cells; almost all logic lives in one big cell:
- **Cell 0** — module docstring + `pip` installs.
- **Cell 2 (Config)** — the only cell you routinely edit: `MODE`, models, datasets, defenses, hyperparameters.
- **Cell 3 (big cell)** — imports, `.env` loading, key loading, corpus builder, model backends
  (`MockModel`/`RealModel`), judge, crash-safe runner, threshold fitting, metrics, **§12b judge-free
  analysis**, plots, `evaluate_model`.
- **Cell 5** — `main()` (runs everything, writes summaries).
- **Cell 6** — `run_selftest()` (mock-only smoke test).
- **Cell 8** — "TEAM STEPS" runbook.

Because Cell 3 is one huge cell, edit it with a small Python script that loads the `.ipynb` JSON and
does targeted string replacements in the cell source, rather than hand-editing JSON. (That's how the
current integrations were applied.)

## Running it
- **`MODE = "mock"`** — verifies the whole pipeline offline in seconds (no GPU, no network, no keys). Run
  this first, then `run_selftest()` → expect `SELFTEST PASSED`.
- **`MODE = "real"`** — real models. Needs a GPU and HF access to gated models. A judge is *not* required to
  run: with `JUDGE_ENABLED=False` the generations are banked and only the judge-free analysis is reported
  (see gotcha below). `MODE` is hard-coded to `"real"` in the config cell.
- Everything **checkpoints per row** to the per-defense `.jsonl`, so a disconnect/rerun resumes.
  Resume keys off `RUN_TAG` — **change `RUN_TAG` for every new seed/config** or a rerun silently
  reuses the previous run's rows.
- Outputs land in `cgp_out/` (gitignored): per-defense `<model>_<tag>_*.jsonl` (incl. generated texts and
  per-token logprobs), `manifest_<tag>.json` (full config + GPU provenance), `separation_<tag>.csv`,
  `separation_by_family_<tag>.csv`, `<model>_<tag>_judgefree.json/.png`, plus `summary.csv` /
  `summary_by_dataset.csv` / `*_analysis.png` once the judge is on.

### VRAM (16 GB cards, e.g. RTX 5070 Ti)
`MODEL_LOAD` sets per-model precision. gemma-3-4b fits in bf16 (~8.6 GB); Llama-3.1-8B does **not**
(~16.1 GB of weights alone) and is set to `4bit`, which needs `bitsandbytes`. Without it,
`device_map="auto"` silently offloads layers to CPU and generation crawls — `_load()` warns when that
happens, and `gpu_report()` warns if a Blackwell (sm_120) card is paired with a pre-cu128 torch build.
**Confound to remember:** 4-bit changes the logprobs `p_ratio_norm` is built from, so a bf16-vs-4bit AUC
gap is not purely a model effect. `manifest_<tag>.json` records the dtype actually used per model.

## Keys / `.env`
Cell 3 calls `_load_dotenv()` before reading keys: it uses `python-dotenv` if installed, else a tiny
built-in parser, and uses `setdefault` so real env vars / Colab Secrets still win. Run the notebook from
this folder so `.env` resolves. On Colab, use the Secrets panel instead (no `.env` there).

## Datasets (equal-sampled)
`DATASETS` lists compiled source keys, each treated as its own dataset; **`N_PER_DATASET` rows are sampled
equally from each**. The 7 sources: `advbench` (goal-only), `advbench_prefill` (forced-prefix),
`safemtdata_multiturn`, `cipherchat_cipher`, `artprompt_orthographic`, `msj_contextwindow`,
`jailbreakbench_general`. Each attack family's rendered prompt (cipher text, ASCII-art cloak, MSJ dialogue,
flattened multi-turn, forced prefix) is mapped to the single text sent to the model; the plaintext goal is
kept only for child-exploitation filtering. See **`compile_datasets.py`** for how each source is built.
If `all_prompts.jsonl` is missing, real mode auto-runs the script; mock/offline uses synthetic placeholders.
Legacy CSV loaders `harmbench` / `strongreject` still work if listed in `DATASETS`. The compiled files cap
at 100/source — raise the `n=` values in `compile_datasets.py` for more.

## Current config defaults
- Run size: `N_PER_DATASET=10`, `N_BENIGN=35` → 70 attack + 35 benign = 105 rows/model (small-model pilot).
- Seeds are **split** into `SAMPLE_SEED` / `GEN_SEED` / `SPLIT_SEED` (all default to `SEED=67`) so a seed
  sweep varies one thing at a time. A single `SEED` used to drive sampling, generation, the split and the
  bootstrap simultaneously, making any spread across seeds unattributable.
- Defenses: `["none", "cgp", "safedecoding"]` (SafeDecoding is a real blended-decoding baseline).
- Threshold: `RATIO_KEY="p_ratio_norm"`, `THRESH_POLICY="balacc"`, held-out train/test split (no leakage).
- Logit-bias hyperparameters (`PROC_PARAMS`): `initial_bias=2.0, max_bias=10.0, ramp_steps=4, grace_period=1`.
- `REFUSAL_PHRASES = ["I cannot", "I am unable", "As an AI", "I must"]`.
- Generation: `max_new_tokens=256, temperature=0.6, top_p=0.9, repetition_penalty=1.15`. Both CGP paths and
  SafeDecoding are seeded per-prompt via `_torch_seed(GEN_SEED, …)`; the base/biased pair shares one stream
  so the comparison stays paired.
- Metrics: ASR / FPR / NAR (both `nonanswer` and `overrefusal`), bootstrap CIs, Mann-Whitney / Wilcoxon
  significance, a "footgun" guard (flags when biasing *raises* ASR), and a per-dataset ASR breakdown.

## Judge-free analysis (section 12b) — valid while the judge is off
`separation_report()` / `stability_report()` / `cost_report()` read **only logprobs**, so they are the part
of the pipeline that produces real numbers today. `judge_free_report()` runs unconditionally in
`evaluate_model()` and writes `<model>_<tag>_judgefree.json`.
- **A1 headline:** AUC of `p_ratio_norm` separating attack from benign, with bootstrap CI and Cliff's delta.
  0.5 = no separation. Reported overall, per prompt type, and per attack family.
- **Controls that matter:** the same AUC for `base_avg` (the ordinary generation's own plausibility) and `H`
  (its entropy). If either separates as well as `p_ratio_norm`, the second biased pass — the entire cost and
  the entire idea — buys nothing. Expect a reviewer to ask exactly this.
- **Stability:** `fit_threshold` infers `direction` from which class has the larger mean; if the classes sit
  close it flips between splits and routing inverts wholesale. `stability_report()` refits over
  `N_SPLIT_REPEATS=200` splits and reports threshold spread + whether direction ever flips.
- **Cost:** measured wall-clock CGP overhead (`t_base`, `t_bias`) and how often the refusal processor
  actually fired (`proc_applied`, `proc_stop_step`) — i.e. whether the biased path did anything at all.

## 🐞 FIXED 2026-07-26: the refusal processor was inert (read before trusting old results)
`RefusalLogitsProcessor.start_ids` was built from `cmap[ph[0]]` — the tokens that *continue* a
refusal phrase after its first character (`'s'`/`'S'` for "as an ai", a bare space for "i cannot").
None of them can **open** a refusal, so the processor could only ever reinforce one the model had
already started on its own. Measured on gemma-3-4b, 105 prompts + a 10-prompt A/B:

| | `max_bias=10` (code) | `max_bias=50` (paper) | after fix, `max_bias=10` |
|---|---|---|---|
| biased text identical to baseline | 70% of 105 rows; 10/10 in probe | 0/10 | 3/10 |
| refusal string, base → biased path | 40/105 → **40/105** | 5/10 → **0/10** | 5/10 → 6/10 |
| mean `p_ratio_norm` | ~0.000 (6 of 7 families) | 0.393 | 0.068 |
| A1 AUC (attack vs benign) | **0.492, CI [0.398, 0.579]** | not measured | not measured |

At `max_bias=10` the junk tokens lost to the natural distribution → **no-op**, `p_ratio_norm ≈ 0`,
nothing for the gate to threshold. At `max_bias=50` they won → the biased path degenerated to
`"I understand you'sssSsSsSsSsS..."`, *destroying* the refusals gemma would otherwise have produced.
The non-zero ratio at 50 measures the plausibility cost of corrupting the output, which tracks
prompt entropy, not harmfulness — and the text the gate would serve is `sSsSsS…`.
Note `MockJudge` scores anything not starting with `"SURE"` as `refused`, so that garbage would be
counted as a successful refusal. **Check whatever produced the paper's Table 1 against this.**

Fix: `_start_ids()` now selects tokens that *begin* a refusal phrase (leading space stripped, so
`"I"` and `" I"` both qualify), cached in `_STARTIDS_CACHE`. Verified: the processor now injects real
refusals (`"…in their community: I cannot fulfill your request…"`).
**The fix is necessary but not sufficient** — the effect is still modest because of the ramp
schedule, not the bug: `grace_period=1, initial_bias=2.0, ramp_steps=4` means only +2.0 at step 1
and `max_bias` at step 5, by which point the sentence is committed. **Sweep `PROC_PARAMS` next.**

Pre-fix data is archived in `cgp_out/prebugfix_broken_processor/` — do not mix it with new runs.
Probe evidence: `cgp_out/probe_maxbias.json` (broken, 10 vs 50), `cgp_out/probe_fixed.json` (fixed).

## Second confound: the benign set makes text-weirdness look like safety
`base_avg` (AUC 0.247) and `H` (0.767) both separated attacks from benign far better than
`p_ratio_norm` did — but per family, plain-English `advbench` was the **weakest** on every signal
(base_avg 0.363) while unreadable `cipherchat_cipher` was **perfect** on all of them (H = 1.000,
mean entropy 3.96 vs 0.306 for benign). With Alpaca as the only benign set, any "is this text
unusual" detector scores well. Add a hard/borderline-benign set (XSTest-style) before concluding
anything about `base_avg`/`H` as controls — and note this confound applies to a *fixed* CGP too.

## Runbook (`tools/`)
```
bash tools/run_all.sh                      # gemma bf16 -> gemma 4bit -> qwen bf16, SERIAL
python tools/run_cgp.py <tag> <model> <quant>   # one model
python tools/preview_auc.py <cgp.jsonl>    # A1 AUC + controls from banked rows, no GPU
python tools/preview_family.py <cgp.jsonl> # per-family AUC for headline vs controls
python tools/check_paths_differ.py <cgp.jsonl>  # SANITY: did the biased path change anything?
python tools/compare_runs.py               # cross-run A1 / stability / cost table
```
**Run `check_paths_differ.py` first on any new run.** If `base_text == bias_text` for most rows the
processor is inert again and every downstream number is meaningless.
Timing on the RTX 5070 Ti: gemma-3-4b ~28 s/row (262k vocab makes every softmax expensive), so
~50 min CGP + ~20 min SafeDecoding per model. Budget ~2.5 h for the three.

## ⚠️ Gotcha: the judge is OFF (`JUDGE_ENABLED = False`)
`get_judge()` returns `NullJudge`, which leaves `judge_*` as `None`; `evaluate_model()` then **skips all
judge-derived output** (ASR / FPR / NAR / defense head-to-head) rather than printing placeholders, and
`summary.csv` comes out empty by design.
Generation is judge-independent and the texts are banked in the `.jsonl`, so: **run the GPU now, judge
later.** Once a key is in `.env`, set `JUDGE_ENABLED = True` and rerun with the *same* `RUN_TAG` — nothing
regenerates, only the missing verdicts are filled.
Do **not** fall back to `MockJudge` on real text: it labels anything not starting with `"SURE"` as refused
(ASR ≈ 0 everywhere), and its refusal strings are exactly the tokens CGP injects, so it scores the defense
on its own output.

## Paper (`yash_aaim (1).pdf`) — being updated
**The code is the source of truth; the paper is rewritten to match the code**, not vice versa (newer models
have shipped since submission, and the pipeline has been substantially upgraded). Known code/paper deltas the
paper text must absorb: model set, `max_bias` (10.0 in code vs 50.0 in paper), refusal-token list, the 7-dataset
expansion, the SafeDecoding baseline, held-out thresholding, CIs/significance, and the resulting Table 1–4 /
threshold-value / abstract-claim changes. Appendix B/C qualitative examples can be regenerated from the saved
`.jsonl` texts. The one blocker for a valid new round is the judge above.
