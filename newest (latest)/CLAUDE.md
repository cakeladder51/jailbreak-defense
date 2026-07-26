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
  (`MockModel`/`RealModel`), judge, crash-safe runner, threshold fitting, metrics, plots, `evaluate_model`.
- **Cell 5** — `main()` (runs everything, writes summaries).
- **Cell 6** — `run_selftest()` (mock-only smoke test).
- **Cell 8** — "TEAM STEPS" runbook.

Because Cell 3 is one huge cell, edit it with a small Python script that loads the `.ipynb` JSON and
does targeted string replacements in the cell source, rather than hand-editing JSON. (That's how the
current integrations were applied.)

## Running it
- **`MODE = "mock"`** — verifies the whole pipeline offline in seconds (no GPU, no network, no keys). Run
  this first, then `run_selftest()` → expect `SELFTEST PASSED`.
- **`MODE = "real"`** — real models + real judge. Needs a GPU, HF access to gated models, and (see gotcha
  below) a working OpenAI judge. `MODE` is currently hard-coded to `"real"` in the config cell.
- Everything **checkpoints per row** to the per-defense `.jsonl`, so a disconnect/rerun resumes.
- Outputs land in `cgp_out/` (gitignored): per-defense `*.jsonl` (incl. generated texts), `summary.csv`,
  `summary_by_dataset.csv`, `*_analysis.png`.

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
- Defenses: `["none", "cgp", "safedecoding"]` (SafeDecoding is a real blended-decoding baseline).
- Threshold: `RATIO_KEY="p_ratio_norm"`, `THRESH_POLICY="balacc"`, held-out train/test split (no leakage).
- Logit-bias hyperparameters (`PROC_PARAMS`): `initial_bias=2.0, max_bias=10.0, ramp_steps=4, grace_period=1`.
- `REFUSAL_PHRASES = ["I cannot", "I am unable", "As an AI", "I must"]`.
- Generation: `max_new_tokens=256, temperature=0.6, top_p=0.9, repetition_penalty=1.15`.
- Metrics: ASR / FPR / NAR (both `nonanswer` and `overrefusal`), bootstrap CIs, Mann-Whitney / Wilcoxon
  significance, a "footgun" guard (flags when biasing *raises* ASR), and a per-dataset ASR breakdown.

## ⚠️ Gotcha: the OpenAI judge is DISABLED
The API key is inactive, so `get_judge()` is stubbed to return the heuristic `MockJudge` even in real mode.
The heuristic is tuned for mock outputs — **on real generations ASR / FPR / NAR are placeholders, not valid.**
Before any real paper-producing run, restore the OpenAI key and re-enable the real judge (uncomment the
original return in `get_judge()`, remove the stub). It logs a warning at runtime while disabled.

## Paper (`yash_aaim (1).pdf`) — being updated
**The code is the source of truth; the paper is rewritten to match the code**, not vice versa (newer models
have shipped since submission, and the pipeline has been substantially upgraded). Known code/paper deltas the
paper text must absorb: model set, `max_bias` (10.0 in code vs 50.0 in paper), refusal-token list, the 7-dataset
expansion, the SafeDecoding baseline, held-out thresholding, CIs/significance, and the resulting Table 1–4 /
threshold-value / abstract-claim changes. Appendix B/C qualitative examples can be regenerated from the saved
`.jsonl` texts. The one blocker for a valid new round is the judge above.
