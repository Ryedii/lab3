import numpy as np
import os
from typing import Any, Generator, List,Tuple,Dict
from threading import Lock
from session import Session
from config import InferenceConfig

class LlamaInterface:
    def __init__(self,config:InferenceConfig) -> None:
        self.max_length = config.max_length
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(config.tokenizer)
        self.sampling_method=config.sampling_method
        self.sampling_value = config.sampling_value
        self.temperature=config.temperature
        self.session=Session.fromConfig(config)
        self.prompt=config.prompt
        self.state:dict[str,Any] = {"code":200,"isEnd":False,"message":""}        
        self.first=True
        self.stop_mp = {"[|Human|]":6,"[|AI|]":5,"<|assistant|>":6,"<|user|>":5,"<|system|>":5}
        self.stop_words = ["<|user|>","<|assistant|>","<|system|>","[|AI|]","[|Human|]"]
        self.model_type = config.model_type
        self.last_output=""
        self.lock = Lock()
        self.reset()
        print("init success")

    def generate_cache(self,prompt:str):
        # 生成缓存
        if len(prompt) == 0 :
            return
        input_ids = np.asarray(self.encode(prompt,add_bos_token=self.first),dtype=np.int64).reshape(1,-1)
        self.first = False
        logits, _, _ = self.session.run(input_ids)
        return self.sample_logits_top_k(logits[0][-1:],self.sampling_value,self.temperature),logits

    def sample_logits_top_k(
        self,
        logits: np.ndarray,
        sampling_value: float = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        logits = logits.astype(np.float32)
        # TODO: 只在概率最高的前 k 个 tokens 中加权采样
        
        return NotImplementedError() # 输出为一个 numpy 数组，表示采样得到的下一个 token id

    
    def format_last_output(self):
        if len(self.last_output) == 0:
            return 
        text_format = self.apply_chat_template([{"role":"assistant","content":self.last_output}])
        self.generate_cache(text_format[len(self.last_output):])
        self.last_output = ""
    
    def predict(self, text):
        if text == "":
            return
        self.format_last_output()
        # TODO: 根据输入文本进行预测
        # hint: self.first 用于判断是否是第一次输入；
        #       self.stop_mp 用于获取停止词的 token id；
        #       可以使用 self.encode 来对输入文本进行编码；
        #       可以使用 self.session.run() 来运行模型，其已封装好了 kv 缓存，
        #       故仅需输入 input_ids 即可；
        #       在遇到 EOS token 或停止词时，需撤销采样结果并回退 kv 缓存状态，
        #       可参考 self.session.rollback() 的相关方法来实现回退；
        
        return NotImplementedError() # 输出为解码后得到的字符串

    def reset(self):
        self.first = True
        self.last_output = ""
        self.session.reset()
        self.generate_cache(self.apply_chat_template(self.prompt))
        
    def getState(self):
        # with self.lock:
        #     return self.state.copy()
        return NotImplementedError()


    def apply_chat_template(self,messages:List[Dict[str,str]]) -> str:
        # TODO: 实现聊天模板，prompt 有三种类型的角色，请参考下面的格式进行实现，输出为
        #
        # 格式参考:
        # https://huggingface.co/docs/transformers/main/en/chat_templating
        #
        # 输入样例:
        # messages = [
        #     {
        #         "role": "system",
        #         "content": "You are a friendly chatbot who always responds in the style of a pirate",
        #     },
        #     {   "role": "user",
        #          "content": "How many helicopters can a human eat in one sitting?"
        #     },
        #     {   "role": "assistant",
        #          "content": "I do not know!"
        #     },
        # ]
        #
        # 输出样例:
        # """
        # <|system|> # 注意这里到下一行需要使用换行符 \n 进行换行
        # You are a friendly chatbot who always responds in the style of a pirate</s>
        # <|user|>
        # How many helicopters can a human eat in one sitting?</s>
        # <|assistant|>
        # I do not know!</s>
        # """
        #
        # 注意，如果最后一条消息是用户消息，则生成的文本以 <|assistant|> 结尾，表示模型需要生成回复
        #
        return NotImplementedError()
    
    def encode(self,text,add_bos_token=False):
        self.tokenizer.add_bos_token = add_bos_token
        return self.tokenizer.encode(text)

def is_stop_word_or_prefix(s: str, stop_words: list) -> int:
    for stop_word in stop_words:
        if s.endswith(stop_word):
            return stop_word
    return ""