from dataclasses import dataclass,field
from typing import Optional,Union,List,Dict
import os
import json

@dataclass
class InferenceConfig:
    tokenizer: str = ""
    hf_model_dir:str = ""
    sampling_method: str = "top_k"
    sampling_value: float = 10 
    temperature: float = 0.7
    max_length:int = 512
    max_input_len:int = 1
    session_type:str="acl"
    acl_mode="rc"
    device:int=0
    prompt:List[Dict[str,str]] = field(default_factory=lambda: [
        {"role":"user","content":"Hey there I am a human that would like to have a conversation with you."},
        {"role":"assistant","content":"Sure, I am happy to answer your questions"},
        {"role":"user","content":"Great, I insist that we take turns."},
        {"role":"assistant","content":"I agree, we should take turns."},
    ])
    model:str=""
    kvcache_method:str = "sliding-window"
    kvcache_fixsize:bool = True
    head_len:int= 32
    recent_len:int = 32
    evict_len:int = 64
    n_layer:int = 22
    format:str='huggingface-tensor'
    # format:str='huggingface-list'
    max_cache_size:int=256
    head_num:int=4
    num_kv_group:int = 8
    head_dim:int=64
    hidden_dim:int=2048
    dtype:str="float16"
    model_type:str="tiny-llama"
    
    def __post_init__(self):
        assert(self.kvcache_method in ["basic","sliding-window",'streamllm','H2O'])
        assert(os.path.isdir(self.hf_model_dir))
        assert(self.session_type in ["acl","onnx"])
        if self.session_type == "onnx":
            self.max_input_len = self.max_length
        self.evict_len = int(min((self.max_cache_size - self.head_len )/2,self.evict_len ))
        self.max_input_len = int(min(self.max_input_len,self.evict_len))
        self.tokenizer = self.hf_model_dir
        model_desc = None
        with open(self.hf_model_dir+"/config.json") as f:
            model_desc = json.load(f)
        self.n_layer = model_desc['num_hidden_layers']
        self.head_num = model_desc['num_key_value_heads']
        self.num_kv_group = int(model_desc['num_attention_heads'] / self.head_num)
        self.hidden_dim = model_desc["hidden_size"]
        self.head_dim = int(self.hidden_dim / model_desc['num_attention_heads'])
        if self.kvcache_method == "streamllm":
            assert(self.head_len+self.evict_len < self.max_cache_size)
        if self.kvcache_method == "H2O":
            self.evict_len = int(min((self.max_cache_size - self.head_len -self.recent_len )/2,self.evict_len ))
            assert(self.head_len+self.recent_len+self.evict_len < self.max_cache_size)
