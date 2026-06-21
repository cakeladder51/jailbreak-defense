import os, sys, time
os.environ["PYTHONUTF8"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import parallel_runner as pr

t0 = time.time()
prompts = pr.load_prompts("all_prompts.jsonl")
os.makedirs("results_parallel", exist_ok=True)

# hyperparameters mirror launch_parallel() defaults
pr.run_worker(
    "Qwen2.5-3B",                                    # model_key
    "Qwen/Qwen2.5-3B-Instruct",                      # model_id
    "results_parallel/results_Qwen2.5-3B.csv",       # out_csv
    prompts,                                         # prompts_by_source
    2.0,    # initial_bias
    50.0,   # max_bias
    4,      # ramp_steps
    1,      # grace_period
    True,   # do_sample
    0.6,    # temperature
    0.9,    # top_p
    10,     # loopback_window
    5,      # top_k_continuations
    1.1,    # repetition_penalty
    True,   # skip_judge
    67,     # sample_size  (x3 sources ~= 201 prompts)
)
print(f"DONE in {time.time()-t0:.0f}s", flush=True)
