import importlib
import sys
import traceback
import random
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig  # if bitsandbytes present; import may fail if not installed
from datasets import load_dataset

# CONFIG
BASE_MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
USE_4BIT = False  # set True only if bitsandbytes is installed & you want 4-bit quant
DEVICE_MAP = "auto"
LOW_CPU_MEM = True

def print_versions():
    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    try:
        import bitsandbytes as bnb
        print("bitsandbytes:", bnb.__version__)
    except Exception:
        print("bitsandbytes: NOT INSTALLED")

def try_load_llava_class_and_model(model_name, quant_config=None, device_map="auto", low_cpu_mem=True):
    candidate_module_paths = [
        "transformers.models.llava.modeling_llava",
        "transformers.models.llava.modeling_llava_for_causal_lm",
        "llava.modeling_llava",
        "modeling_llava",
    ]
    candidate_class_names = [
        "LlavaForCausalLM",
        "LlavaModelForCausalLM",
        "LlavaForConditionalGeneration",
        "LlavaModel",
        "LlavaForVision2Seq",
    ]

    last_exc = None
    for mod_path in candidate_module_paths:
        try:
            module = importlib.import_module(mod_path)
        except Exception as e:
            last_exc = e
            continue

        for cls_name in candidate_class_names:
            ModelClass = getattr(module, cls_name, None)
            if ModelClass is None:
                continue

            # Try strategy sequence:
            # 1) If quant_config provided -> try direct (fast path)
            # 2) If ValueError complaining about dispatch -> retry with llm_int8_enable_fp32_cpu_offload + device_map="auto"
            # 3) If still failing -> fallback to no-quant (float16)
            try:
                print(f"Trying to load {cls_name} with device_map={device_map} (quant_config={'yes' if quant_config else 'no'})...")
                return _attempt_from_pretrained(ModelClass, model_name, quant_config, device_map, low_cpu_mem, extra_kwargs={})
            except Exception as e:
                last_exc = e
                tb = traceback.format_exc()
                print(f"Initial attempt with {cls_name} failed: {e}\n{tb}")

                # If message suggests offload, try offload route (only if quant_config not None)
                msg = str(e).lower()
                if quant_config is not None and ("offload" in msg or "dispatched on the cpu" in msg or "some modules are dispatched" in msg):
                    try:
                        print("Retrying with llm_int8_enable_fp32_cpu_offload=True and device_map='auto'...")
                        return ModelClass.from_pretrained(
                            model_name,
                            quantization_config=quant_config,
                            device_map="auto",
                            trust_remote_code=True,
                            low_cpu_mem_usage=low_cpu_mem,
                            llm_int8_enable_fp32_cpu_offload=True,
                        )
                    except Exception as e2:
                        last_exc = e2
                        tb2 = traceback.format_exc()
                        print(f"Retry with offload failed: {e2}\n{tb2}")

                # Final fallback: try without quantization (float16)
                try:
                    print("Retrying without quantization (float16) as a fallback...")
                    return ModelClass.from_pretrained(
                        model_name,
                        device_map=device_map,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        trust_remote_code=True,
                        low_cpu_mem_usage=low_cpu_mem,
                    )
                except Exception as e3:
                    last_exc = e3
                    tb3 = traceback.format_exc()
                    print(f"Fallback without quantization also failed: {e3}\n{tb3}")
                    continue

    raise RuntimeError("Tried candidate Llava classes but all failed. Last exception:\n" + (str(last_exc) if last_exc is not None else "None"))

def _attempt_from_pretrained(ModelClass, model_name, quant_config, device_map, low_cpu_mem, extra_kwargs):
    """Helper to call from_pretrained with given kwargs and bubble exceptions."""
    try:
        if quant_config is not None:
            return ModelClass.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=low_cpu_mem,
                **extra_kwargs,
            )
        else:
            return ModelClass.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=low_cpu_mem,
                **extra_kwargs,
            )
    except Exception as e:
        raise



def load_model_and_tokenizer(model_name, use_4bit=True, device_map="auto", low_cpu_mem=True):
    # Load config + tokenizer
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = None
    if use_4bit:
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        except Exception as e:
            print("Could not create BitsAndBytesConfig:", e)
            quant_config = None

    # If LlavaConfig detected: try repo model classes (with fallback paths)
    cfg_name = config.__class__.__name__.lower()
    if "llava" in cfg_name:
        model = try_load_llava_class_and_model(model_name, quant_config=quant_config, device_map=device_map, low_cpu_mem=low_cpu_mem)
    else:
        # generic fallback to AutoModelForCausalLM
        if quant_config is not None:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quant_config,
                    device_map=device_map,
                    trust_remote_code=True,
                    low_cpu_mem_usage=low_cpu_mem,
                    llm_int8_enable_fp32_cpu_offload=True,  # safe to include
                )
            except Exception as e:
                print("AutoModelForCausalLM with quant failed, retrying without quant:", e)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device_map,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=low_cpu_mem,
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=low_cpu_mem,
            )

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    model.config.use_cache = False
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    print("Model loaded. Sample module names (first 50):")
    for i, (n, m) in enumerate(model.named_modules()):
        if i >= 50:
            break
        print(i, n)
    print("Done.")
