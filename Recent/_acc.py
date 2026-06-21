import os
from huggingface_hub import HfApi, model_info
tok = os.environ.get("HF_TOKEN", "")
try:
    who = HfApi().whoami(token=tok)
    print("token user:", who.get("name"))
except Exception as e:
    print("whoami err:", str(e)[:120])
for m in ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
          "microsoft/Phi-3.5-mini-instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"]:
    try:
        model_info(m, token=tok)
        print("ACCESS OK:", m)
    except Exception as e:
        print("NO:", m, "->", str(e)[:90])
