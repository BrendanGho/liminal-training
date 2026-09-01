import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USER_ID = os.getenv("HF_USER_ID", "")

VLLM_N_GPUS = int(os.getenv("VLLM_N_GPUS", 0))
# Must be >= the --lora-rank used in training (default 64), or vLLM refuses
# to load the adapter at evaluation time.
VLLM_MAX_LORA_RANK = int(os.getenv("VLLM_MAX_LORA_RANK", 64))
VLLM_MAX_NUM_SEQS = int(os.getenv("VLLM_MAX_NUM_SEQS", 512))
