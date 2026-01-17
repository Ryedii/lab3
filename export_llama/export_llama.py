import argparse
import importlib
import torch
import os
from transformers import LlamaForCausalLM, LlamaTokenizer


def export_onnx(base_model,out_path,quant_cfg_path,act_path):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    device = "cpu"
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    ).to(device)
    model_cfg=model.model.config
    spec = importlib.util.spec_from_file_location("quant_cfg_module", quant_cfg_path)
    quant_cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quant_cfg_module)
    quantize_cfg = quant_cfg_module.get(model_cfg,act_path)
    from quantize import quantize
    quantize(model,quantize_cfg)
    
    input_names = ["input_ids", "attention_mask", "position_ids","past_key_values"]
    output_names = ["logits","out_key_values","attn_scores"]
    
    batch_size,seq_len,kv_len=1,1,1024 # 请勿修改

    # TODO: 构造导出假输入，需要和模型实际输入对应上
    # input_ids 形状为 (batch_size, sequence_length)
    # attention_mask 形状为 (batch_size, all_sequence_length(what's that?))
    # position_ids 形状为 (batch_size, sequence_length)
    # past_key_values 形状为 (n_layers, 2, batch_size, n_heads, kv_len, head_dim)
    
    input_args = (
        # TODO: 填写下面的输入
        # input_ids: torch.LongTensor = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,          # 这里填 None
        # labels: Optional[torch.LongTensor] = None,                  # 这里填 None
        # use_cache: Optional[bool] = None,                           # 这里填 True 才能输出 past_key_values
        # output_attentions: Optional[bool] = None,                   # 这里填 True
    )
    
    dynamic_axes = {
        # TODO: 根据上面构造的输入输出，填写 dynamic_axes
        # 格式为:
        # name: { axis_index: "axis_name", ... }
        # 如: 
        # "input_ids": { 0: "batch_size", 1: "seq_length" }
        #
        # hint: 格外注意 past_key_values，哪些维度是可变的？
    }


    model.eval()
    raise NotImplementedError("Please complete the TODOs in export_onnx function before running.")
    torch.onnx.export(
        model,
        f=out_path,
        args=input_args,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
        export_params=True,
    )

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m",
        type=str, 
        default="./model/TinyLlama-1.1B-Chat-v1.0", 
        help="transformers model"
    )
    parser.add_argument(
        "--output","-o",
        type=str,
        default="./model/export_out/TinyLlama-chat-v1.0-quant.onnx",
        help="where to save onnx model",
    )
    parser.add_argument(
        "--act-path","-a",
        type=str,
        default="./act/TinyLlama-chat-v1.0-act.pt",
        help="path to act_scales",
    )
    parser.add_argument(
        "--quant","-q",
        type=str,
        default="./config/w8x8.py",
        help="path to quant config",
    )
    args = parser.parse_args()
    export_onnx(args.model,args.output,args.quant,args.act_path)