from config import InferenceConfig
from kvcache import KVCache
import numpy as np
from typing import List
import time
import sys
class Session:
	def __init__(self,config:InferenceConfig) -> None:
		self.kvCache = KVCache.create(config)
		self.max_len = config.max_input_len

	def run(self,input_ids:np.ndarray):
		pass
	
	@staticmethod
	def fromConfig(config:InferenceConfig) -> 'Session':
		if config.session_type == "onnx":
			return NotImplementedError()
		elif config.session_type=='acl':
			return AclSession(config)
		else:
			return None
	
	def reset(self):
		self.kvCache.reset()

	def rollback(self,seq_len):
		self.kvCache.rollback(seq_len)

	def evict(self,space_need):
		self.kvCache.evict(space_need)
	
class OnnxSession(Session):
	def __init__(self,config:InferenceConfig)->None:
		raise NotImplementedError()

	def run(self,input_ids:np.ndarray):
		return NotImplementedError()

class AclSession(Session):
	context = None
	def __init__(self,config:InferenceConfig)->None:
		super().__init__(config)
		from engine import ACLModel,initResource
		self.context = initResource(config.device)
		self.model = ACLModel(config.model,context=self.context,mode=config.acl_mode)
		self.input_ids = np.zeros((1,self.max_len),dtype=np.int64)
		if config.acl_mode == 'rc':
			self.input_ids,_,_,self.kvCache.kvCache = self.model.getInputs()

	def run(self,input_ids:np.ndarray):
		seq_len=input_ids.shape[-1]
		l,r,result = 0,self.max_len,None

		def list_to_huggingface_tensor_kv(lst:List[np.ndarray])->np.ndarray:
			assert len(lst) % 2 == 0
			n_layer = len(lst) // 2
			keys = []
			values = []

			for i in range(n_layer):
				key = lst[2 * i]
				value = lst[1 + 2 * i]
				keys.append(key)
				values.append(value)
			
			# keys: [22, 1, 4, 1025, 64]
			# values: [22, 1, 4, 1025, 64]
			keys = np.stack(keys, axis=0)
			values = np.stack(values, axis=0)
			return np.stack([keys, values], axis=1)  # (22, 2, 1, 4, 1025, 64)

		def list_to_huggingface_list_kv(lst:List[np.ndarray]):
			assert len(lst) % 2 == 0
			n_layer = len(lst) // 2
			kv = []

			for i in range(n_layer):
				key = lst[2 * i]
				value = lst[1 + 2 * i]
				kv.append([key, value])
			
			# keys: [22, 1, 4, 1025, 64]
			# values: [22, 1, 4, 1025, 64]
			return kv  # [22, [2, (1, 4, 1025, 64) ] ]

		def huggingface_tensor_to_list_kv(t):
			kv = []
			n_layer = t.shape[0]
			for i in range(n_layer):
				kv.append([t[i,0], t[i, 1]])
			return kv

		while l < seq_len:
			r = min(seq_len,r)
			self.input_ids[:,:r-l] = input_ids[:,l:r]
			cache, mask, pos_ids = self.kvCache.getInputs(self.max_len)
			result:List[np.ndarray] = self.model.inference([self.input_ids,mask,pos_ids,cache])
			logits = result[0]
			past_key_values = list_to_huggingface_tensor_kv(result[1:45])
			attn_scores = np.stack(result[45:67], axis=0)

			self.kvCache.update(r-l, past_key_values, attn_scores)
			l , r = l + self.max_len , r + self.max_len
		return logits, past_key_values, attn_scores