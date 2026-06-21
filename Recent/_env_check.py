import importlib, os

print("== files ==")
for f in ["all_prompts.jsonl", "parallel_runner.py", "results_parallel/results_Llama-3.1-8B.csv"]:
    print(f"  {f}: {'EXISTS' if os.path.exists(f) else 'missing'}")

print("== deps ==")
for m in ["torch", "transformers", "pandas", "matplotlib", "filelock", "accelerate"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m}: {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m}: MISSING ({type(e).__name__})")

print("== cuda ==")
try:
    import torch
    print("  cuda available:", torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name} {round(p.total_memory/1e9,1)} GB")
except Exception as e:
    print("  torch err:", e)
