import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from config import InferenceConfig
from inference import LlamaInterface

def main(engine:LlamaInterface):
    while True:
        line = input()
        print(engine.predict(line))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv_size", type=int, default=1024)
    parser.add_argument(
        "--sampling_value",type=float,default=10,
        help="the value of p."
    )
    parser.add_argument(
        "--temperature",type=float,default=0.7,
        help="sampling temperature."
    )
    parser.add_argument(
        "--hf-dir", type=str, default="/root/exp/model/TinyLlama", 
        help="path to huggingface model dir"
    )
    parser.add_argument(
        "--model", type=str, default="/root/exp/model/TinyLlama-chat-v1.0-quant-w8x8.om", 
        help="path to onnx or om model"
    )
    args = parser.parse_args()
    cfg = InferenceConfig(
        hf_model_dir=args.hf_dir,
        model=args.model,
        max_cache_size=args.kv_size,
        sampling_method="top_k",
        sampling_value=args.sampling_value,
        temperature=args.temperature,
        session_type="acl",
    )
    engine = LlamaInterface(cfg)
    main(engine)