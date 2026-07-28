# Resume here — state as of 2026-07-28 ~01:00

Working notes for picking this back up. Delete or rewrite when the next milestone lands.

## Where things stand

Two things happened on 2026-07-27, in this order:

1. **A reporting error was found and corrected.** `_auc()` scores with attacks as the positive
   class, but CGP predicts attacks have the *smaller* gap ("gap below threshold → serve the
   refusal"), so AUC < 0.5 is the **intended** direction and `fit_threshold` infers that sign
   itself. Reading "< 0.5" as inversion had turned a working signal into a reported failure
   across `docs/`, `CLAUDE.md`, commit `69f7a33` and the supervisor artifact. All now carry dated
   corrections; the original text was left intact rather than silently overwritten.
2. **The hard benign set landed and vindicated CGP's premise.** See below.

## The result that matters

`BENIGN_SOURCES = {"alpaca": 35, "xstest_safe": 35}` — Alpaca is easy benign, XSTest-safe is hard
(prompts that look harmful but are not). Oriented AUC of attacks against **each** benign set:

| signal | gemma easy→hard | qwen easy→hard | verdict |
|---|---|---|---|
| `bias_is_refusal` | 0.764 → **0.764** (+0.00) | 0.800 → **0.771** (−0.03) | holds |
| `refusal_then_gap` | 0.692 → **0.738** (+0.05) | 0.813 → **0.776** (−0.04) | holds |
| `p_ratio_norm` | 0.525 → **0.637** (+0.11) | 0.665 → **0.654** (−0.01) | holds |
| `base_avg` *(control)* | 0.753 → 0.653 (**−0.10**) | 0.654 → 0.541 (**−0.11**) | collapses |
| `H` *(control)* | 0.767 → 0.602 (**−0.17**) | 0.674 → 0.579 (**−0.10**) | collapses |

**The controls were the ones riding the benign-set confound, not CGP.** They only beat CGP on
Alpaca — the comparison that could not distinguish "detects attacks" from "detects unusual text".
Mean entropy: attacks 0.912, Alpaca 0.306, XSTest 0.390 — `H` was separating on register.

Over-refusal (biased-path refusal rate on benign) is the cost side, and it is model-dependent:
gemma 0.03 on both benign sets (against 0.56 on attacks); qwen 0.23 Alpaca / **0.29 XSTest**
(against 0.83). Part of qwen's is its own exaggerated safety — its *baseline* path already
refuses 11% of XSTest vs 6% of Alpaca.

## ⏳ First thing tomorrow

**phi-3-mini's hard-benign run (`phi3_hb`) was still going when we stopped** (~4/140 at 00:44).
It resumes for free — rows already banked are skipped:

```bash
bash tools/run_hardbenign.sh          # re-runs all three; gemma_hb and qwen_hb are already
                                      # complete so they no-op, phi3_hb picks up where it left off
```

Then:

```bash
python tools/check_paths_differ.py cgp_out/phi-3-mini_phi3_hb_cgp.jsonl   # ALWAYS first
python tools/preview_auc.py       cgp_out/phi-3-mini_phi3_hb_cgp.jsonl   # easy-vs-hard table
python tools/compare_runs.py                                             # all three side by side
```

phi is the model where `p_ratio_norm` was strongest (0.798 ex-cipher) and `bias_is_refusal`
weakest (0.729), so it is the one that decides whether `refusal_then_gap` is genuinely the better
default or just better on two of three.

## Then, in order

1. **Decide whether to switch `RATIO_KEY`.** `bias_is_refusal` and `refusal_then_gap` are banked
   and reported but **`RATIO_KEY` still defaults to `p_ratio_norm`** — deliberately. Switching it
   is a real change to what CGP thresholds and needs all three models in. If it goes ahead:
   `route()`/`fit_threshold()` need no changes (they are key-agnostic), but the threshold on a
   near-binary signal is degenerate, so `refusal_then_gap` is the sane choice over
   `bias_is_refusal` alone.
2. **Re-sweep `max_bias` per model.** The old sweep was scored with the sign error; corrected and
   ex-cipher it runs 0.608 → 0.750 → 0.792 as `max_bias` goes 10 → 15 → 25, i.e. *monotonically
   better*, the opposite of the shipped conclusion. `tools/sweep_proc.py` needs its metric changed
   to oriented AUC on the hard benign set before it is trusted again. gemma and qwen sit in
   opposite regimes at the same value, so this should produce a per-model number.
3. **Investigate `advbench_prefill`.** Direction-aware, it is **0.421 on gemma** — genuinely
   backwards — and 0.63–0.80 elsewhere. Forced-prefix is the paper's headline attack and it is the
   weakest plain family on every model.
4. **Two families resist everything**: `cipherchat_cipher` (biased-path refusal rate 0.00 on
   gemma/phi — the models do not recognise ciphered requests as harmful, so there is nothing to
   read; CGP's ceiling is the model's own refusal tendency) and `safemtdata_multiturn` (~0.5 on
   every signal, unexplained).
5. **Judge pass** when a valid `OPENAI_API_KEY` exists — the current one is invalid. Set
   `JUDGE_ENABLED = True`, rerun with the **same** `RUN_TAG`; nothing regenerates.
   `tools/predict_judge.py` records the expected outcome (~+2 to +4pp refusals on harmful through
   the gate) so the judge can be checked against it.

## Gotchas that will bite

- **Change `RUN_TAG` for every new config.** Resume keys off it; a rerun silently reuses rows.
- **`tools/check_paths_differ.py` first, every time.** If `base_text == bias_text` dominates, the
  processor is inert and everything downstream is meaningless.
- **Read oriented AUC + direction, never raw AUC alone.** That mistake cost a day.
- **The GPU ran at half clock** (1590 of 3090 MHz, 84 W of 300 W, no throttle reason active) with
  NVIDIA Broadcast and Parsec resident — 29 s/row instead of 13. Output is byte-identical, only
  wall clock differs; closing those apps roughly halves run time.
- **Do not run the judge with a bad key** — fixed now (failed calls return `None`), but confirm the
  key works before committing to a full pass.
