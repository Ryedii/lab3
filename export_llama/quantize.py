import torch
from torch import nn,Tensor
from typing import Optional,List,Tuple
from torch.onnx.symbolic_helper import parse_args

class MatMulInteger(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:torch.Tensor,weight_t:torch.Tensor):               
        res = torch.matmul(x.to(dtype=torch.float32),weight_t.to(torch.float32))
        return res

    @staticmethod
    @parse_args("v","v")
    def symbolic(g:torch._C.Graph, x:torch.Tensor,weight_t:torch.Tensor):
        return g.op("MatMulInteger", x,weight_t)

matmulInteger = MatMulInteger.apply

def quantize_mat(mat:Tensor)-> Tuple[Tensor,Tensor]:
    # max_val = torch.max(torch.abs(mat),dim=-1)[0]
    # mat =  (mat * (127 / max_val)[...,None]).to(dtype=torch.int8)
    max_val = (torch.max(torch.abs(mat),dim=-1)[0] / 127.0).to(dtype=mat.dtype)
    mat =  (mat / max_val[...,None]).to(dtype=torch.int8)
    return mat, max_val

def dequantize_mat(mat:Tensor,max_val:Tensor):
    return torch.mul(mat,max_val.unsqueeze(-1))

def decomposition(mat:Tensor,unq_idx:Tensor,t:Tensor) -> Tuple[Tensor,Tensor,Tensor,Tensor]:
    return mat.mul(t.to(dtype=mat.dtype)),mat[...,unq_idx]
    mat=mat.clone()
    mat_unq = mat[...,unq_idx]
    if mat.dim() == 3:
        mat[:,:,unq_idx] = 0
    elif mat.dim() == 4:
        mat[:,:,:,unq_idx] = 0
    elif mat.dim() == 2:
        mat[:,unq_idx] = 0
    return mat,mat_unq

def get_unq_idx_topk(mat:Tensor,k:int=64):
    idx=torch.topk(mat.view(-1,mat.shape[-1]).abs().max(dim=-2)[0],k,dim=-1)[1]
    t = torch.ones((mat.shape[-1]),dtype=mat.dtype,device=mat.device)
    t = t.clone()
    t[idx] = 0
    return idx,t

def get_unq_idx_thres(mat:Tensor,threshold:float=6.0):
    k = mat.view(-1,mat.shape[-1]).abs().max(dim=-2)[0] >= threshold
    return k.nonzero().view(-1), k

def qMatmul(x_q:Tensor,x_max:Tensor,weight_q:Tensor,w_max:Tensor,dtype):
    res_q = matmulInteger(x_q , weight_q)
    mx = nn.functional.linear(x_max.unsqueeze(-1),w_max.unsqueeze(-1))
    res = torch.mul(res_q.to(device=mx.device,dtype=torch.float32), mx.to(torch.float32) ).to(dtype=dtype)  
    # res = torch.mul((res_q.to(device=mx.device,dtype=torch.float32) / (127.0*127.0)).to(torch.float16), mx )  
    return res

class W8Linear(nn.Module):
    def __init__(self, origin_weight:Tensor, bias: Optional[Tensor] = None,act_max:Optional[Tensor] = None,alpha=32):
        raise NotImplementedError

    def forward(self,x:Tensor) -> Tensor:
        return NotImplementedError

# act_max for smooth 
class W8X8Linear(nn.Module):
    def __init__(self, ori_w:Tensor, bias: Optional[Tensor] = None,act_max:Optional[Tensor] = None,alpha=32):
        super().__init__()
        self.bias = None if bias is None else nn.Parameter(bias,requires_grad=False)
        self.dtype = ori_w.dtype
        self.alpha = alpha
        self.scales = None
        if act_max is not None:
            act_max = act_max.to(ori_w.device)
            self.scales = (act_max.pow(alpha) / ori_w.abs().max(dim=0)[0].pow(1 - alpha)).clamp(min=1e-5).to(dtype=ori_w.dtype)
            self.scales = nn.Parameter(self.scales,requires_grad=False)
            ori_w = ori_w.detach().mul(self.scales)
        self.weight_q,self.max_val = quantize_mat(ori_w.detach())
        self.weight_q = nn.Parameter(self.weight_q.t(),requires_grad=False)
        self.max_val = nn.Parameter(self.max_val,requires_grad=False)

    def forward(self,x:Tensor) -> Tensor:
        # TODO: W8X8 前向传播
        # hint: 可以使用上面已经实现好的函数 qMatmul() 等
        # 对输入x进行per-token量化（在最后一个维度上量化）
        # x的形状通常是 (batch, seq_len, hidden_dim) 或 (batch*seq_len, hidden_dim)
        x_flat = x.view(-1, x.shape[-1])  # 展平batch和seq维度
        
        # 计算每个token的最大绝对值（在hidden_dim维度上）
        x_max_abs = x_flat.abs().max(dim=-1, keepdim=True)[0]  # (num_tokens, 1)
        # 添加小的epsilon避免除零错误
        eps = 1e-8
        x_max_abs = x_max_abs.clamp(min=eps)
        # 计算缩放因子（类似quantize_mat的逻辑）
        x_max = (x_max_abs / 127.0).to(dtype=self.dtype).squeeze(-1)  # (num_tokens,)
        # 量化输入
        x_q = (x_flat / x_max.unsqueeze(-1)).to(dtype=torch.int8)  # (num_tokens, hidden_dim)
        
        # 使用qMatmul进行量化矩阵乘法
        # 注意：weight_q已经转置了，所以直接用
        output = qMatmul(x_q, x_max, self.weight_q, self.max_val, self.dtype)
        
        # 恢复原始形状
        if x.dim() > 2:
            output = output.view(x.shape[:-1] + (-1,))
        
        # 添加bias（如果有）
        if self.bias is not None:
            output = output + self.bias
        
        return output

# static decomposition
class W8SDLinear(nn.Module):
    def __init__(self, origin_weight:Tensor, bias: Optional[Tensor] = None,act_max:Optional[Tensor] = None,alpha=32):
        raise NotImplementedError

    def forward(self,x:Tensor) -> Tensor:
        return NotImplementedError
    
class W8DXLinear(nn.Module):
    def __init__(self, origin_weight:Tensor, bias: Optional[Tensor] = None,act_max:Optional[Tensor] = None,alpha=32):
        raise NotImplementedError

    def forward(self,x:Tensor) -> Tensor:
        return NotImplementedError


quant_cls = {
    "W8":W8Linear,
    "W8X8":W8X8Linear,
    "W8SD":W8SDLinear,
    "W8DX":W8DXLinear
}         
        
def replace_linear_modules(module:nn.Module,prefix:str,act_scales,cfg):
    for name, child in module.named_children():
        fullname = (prefix + '.' + name) if prefix != '' else name
        if isinstance(child, nn.Linear):
            strs = fullname.split(".")
            # fullname: model.layers.21.self_attn.q_proj layer_name: 21.q_proj; name: q_proj
            # fullname: lm_head; layer_name: 21.q_proj; name: q_proj;
            layer_name = (strs[-3] + "." + strs[-1]) if len(strs) > 2 else strs[-1]
            if layer_name not in cfg:
                continue
            act_scale = None if act_scales is None or 'act_scale' not in cfg[layer_name] else act_scales[fullname]
            alpha = 32 if 'alpha' not in cfg[layer_name] else cfg[layer_name]['alpha']
            setattr(module, name,quant_cls[cfg[layer_name]['type']]
                    (child.weight,child.bias,act_max=act_scale,alpha=alpha))
        else:
            replace_linear_modules(child,fullname,act_scales,cfg)

def quantize(model:nn.Module,cfg={}):
    act_scales = None
    if 'act_scales_path' in cfg:
        act_scales = torch.load(cfg['act_scales_path'])
        if 'smooth' in cfg:
            from smooth import smooth_lm
            alpha = 0.85 if "alpha" not in cfg else cfg["alpha"]
            smooth_lm(model, act_scales, alpha)
    replace_linear_modules(model,'',act_scales,cfg)