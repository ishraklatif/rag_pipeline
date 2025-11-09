"""
Loads the foundation LLM defined in config.py.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_community.llms import HuggingFacePipeline

from config import LLM_MODEL_NAME


def get_llm(temperature: float = 0.3, max_new_tokens: int = 512) -> HuggingFacePipeline:
    """Load LLM specified in config.py
    """
    print(f"\n⚡ Loading model: {LLM_MODEL_NAME}")

    # Detect hardware
    if torch.backends.mps.is_available():
        device_map = "auto"
        torch_dtype = torch.float16
        print("Using Apple Metal (MPS) backend")
    elif torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = torch.float16
        print("Using CUDA backend")
    else:
        device_map = None
        torch_dtype = torch.float32
        print("Using CPU backend")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

    # Build text generation pipeline (no manual device argument)
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
    )

    return HuggingFacePipeline(pipeline=gen_pipe)
