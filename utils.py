import torch, os, random
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelling_llama import LlamaForCausalLM
from modelling_qwen2p5 import Qwen2ForCausalLM
from modelling_qwen3 import Qwen3ForCausalLM


def load_model(model_name, device_id):
    if model_name == 'llama7b':
        model_path = '/models/Llama-2-7b-chat-hf' #'meta-llama/Llama-2-7b-chat-hf'

        tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side="left", padding_side="left")
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device_id  # or 'auto' if you want auto GPU placement
        )
    
    elif model_name == 'llama13b':
        model_path =  '/models/Llama-2-13b-chat-hf' #'meta-llama/Llama-2-13b-chat-hf'
        tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side="left", padding_side="left")
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device_id  # or 'auto' if you want auto GPU placement
        )
    
    elif model_name == 'qwen7b_inst':
        model_path = f'/models/Qwen2.5-7B-Instruct'
        model = Qwen2ForCausalLM.from_pretrained(   
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map= device_id
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side="left", padding_side="left")
    
    elif model_name == 'qwen32b_inst':
        model_path = f'/models/Qwen2.5-32B-Instruct-bnb-4bit/'
        model = Qwen2ForCausalLM.from_pretrained(   
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map= device_id
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side="left", padding_side="left")

    elif model_name == 'qwen14b_inst':
        model_path = f'/models/Qwen3-14B-Instruct'
        model = Qwen3ForCausalLM.from_pretrained(   
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map= device_id
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side="left", padding_side="left")

    else:
        assert False

    model = model.eval()

    return tokenizer, model


def load_val_model(val_device_id):
    val_cls = AutoModelForCausalLM.from_pretrained("/models/HarmBench_Mistral-7b-val-cls", torch_dtype=torch.bfloat16, device_map=val_device_id)
    
    val_tokenizer = AutoTokenizer.from_pretrained("/models/HarmBench_Mistral-7b-val-cls", use_fast=False, truncation_side="left", padding_side="left")
    val_cls = val_cls.eval()
    
    return val_tokenizer, val_cls


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

