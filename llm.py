"""
Loads the foundation LLM defined in config.py.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline   # updated import

from config import LLM_MODEL_NAME

# Suppress the USER_AGENT warning from LangChain web loaders
os.environ.setdefault("USER_AGENT", "rag-pipeline/1.0")


def get_llm(temperature: float = 0.3, max_new_tokens: int = 512) -> HuggingFacePipeline:
    """Load LLM specified in config.py"""
    print(f"\n⚡ Loading model: {LLM_MODEL_NAME}")

    # Detect hardware
    if torch.backends.mps.is_available():
        device_map = "auto"
        dtype = torch.float16
        print("Using Apple Metal (MPS) backend")
    elif torch.cuda.is_available():
        device_map = "auto"
        dtype = torch.float16
        print("Using CUDA backend")
    else:
        device_map = None
        dtype = torch.float32
        print("Using CPU backend")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        device_map=device_map,
        dtype=dtype,               # replaces deprecated torch_dtype
    )

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        return_full_text=False,
    )

    return HuggingFacePipeline(pipeline=gen_pipe)
