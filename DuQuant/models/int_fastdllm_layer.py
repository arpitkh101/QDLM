import torch
from torch import nn
from typing import Optional, Tuple, List
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
import torch.nn.functional as F
from quantize.du_norm import DuLlamaRMSNorm
from collections import OrderedDict
import math
from transformers.activations import ACT2FN
import copy
from models.transformation import *
from models.configuration_fastdllm import Fast_dLLM_QwenConfig
from models.modelling_fastdllm import apply_rotary_pos_emb, repeat_kv

class QuantFastdLLMMLP(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        args=None,
    ):
        super().__init__()
        
        self.gate_proj = QuantLinear(org_module.gate_proj,
                                           args.gate_weight_quant_params,
                                           args.gate_act_quant_params)
        self.down_proj = QuantLinear(org_module.down_proj,
                                           args.down_weight_quant_params,
                                           args.down_act_quant_params)
        self.up_proj = QuantLinear(org_module.up_proj,
                                           args.up_weight_quant_params,
                                           args.up_act_quant_params)
        self.act_fn = ACT2FN[hidden_act]
        self.init_duquant_params = torch.tensor(0) if args.gate_weight_quant_params['quant_method'] == 'duquant' else torch.tensor(1)

    def forward(self, x):
        if not self.init_duquant_params:
            self.init_duquant_params = torch.tensor(1)
            act = self.act_fn(self.gate_proj(x))
            self.up_proj.copy_quantizers_duquant_params(self.gate_proj)
            mul = act * self.up_proj(x)
            return self.down_proj(mul)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class QuantFastdLLMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, 
                 org_module: nn.Module,
                 config: Fast_dLLM_QwenConfig,
                 args=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.k_proj = QuantLinear(
            org_module.k_proj,
            args.k_weight_quant_params,
            args.k_act_quant_params,
        )
        self.v_proj = QuantLinear(
            org_module.v_proj,
            args.v_weight_quant_params,
            args.v_act_quant_params,
        )
        self.q_proj = QuantLinear(
            org_module.q_proj,
            args.q_weight_quant_params,
            args.q_act_quant_params,
        )
        self.o_proj = QuantLinear(
            org_module.o_proj, args.o_weight_quant_params, args.o_act_quant_params
        )
        self.qkt_matmul = QuantMatMul(
            args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul, rotate=None
        )
        self.pv_matmul = QuantMatMul(
            args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul, rotate=None
        )

        self.use_weight_quant = False
        self.use_act_quant = False
        self.init_duquant_params = torch.tensor(0) if args.gate_weight_quant_params['quant_method'] == 'duquant' else torch.tensor(1)
        
        self.sliding_window = config.sliding_window if hasattr(config, "layer_types") and hasattr(org_module, "layer_idx") and config.layer_types[org_module.layer_idx] == "sliding_attention" else None
        self.layer_idx = org_module.layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        update_past_key_values: Optional[bool] = False,
        block_past_key_values: Optional[Tuple[torch.Tensor]] = None,
        replace_position: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        if not self.init_duquant_params:
            self.k_proj.copy_quantizers_duquant_params(self.q_proj)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        if not self.init_duquant_params:
            self.v_proj.copy_quantizers_duquant_params(self.q_proj)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        cos = cos.to(query_states.device)
        sin = sin.to(query_states.device)
        
        # RoPE application
        # Note: The original code handles training vs inference split for RoPE. 
        # Assuming inference here as quantization is usually post-training or for inference.
        # print(f"DEBUG: q.shape={query_states.shape}, k.shape={key_states.shape}, cos.shape={cos.shape}")
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Block Cache Logic
        if block_past_key_values is not None:
            if len(block_past_key_values) <= self.layer_idx:
                # This part in original code uses Cache object's update. 
                # Since we are likely using a simplified cache or tuple in this context, we might need to adapt.
                # However, for DuQuant main loop, we might not be hitting this complex generation path.
                # Let's assume standard cache behavior for now or try to support what's passed.
                # If block_past_key_values is a Cache object, we can call update.
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = block_past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                block_cache_key_states = block_past_key_values[self.layer_idx][0]
                block_cache_value_states = block_past_key_values[self.layer_idx][1]
                
                block_cache_key_states[:, :, replace_position:replace_position+key_states.shape[2]] = key_states
                block_cache_value_states[:, :, replace_position:replace_position+value_states.shape[2]] = value_states
                key_states = block_cache_key_states
                value_states = block_cache_value_states

        if past_key_value is not None:
            if update_past_key_values:
                 # Assuming past_key_value is a Cache object if update_past_key_values is True
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            elif len(past_key_value) > self.layer_idx:
                 # Tuple fallback
                key_states = torch.cat((past_key_value[self.layer_idx][0], key_states), dim=-2)
                value_states = torch.cat((past_key_value[self.layer_idx][1], value_states), dim=-2)

        # Attention Calculation with Explicit MatMul for Quantization
        
        # query_states: [bsz, num_heads, q_len, head_dim]
        # key_states: [bsz, num_key_value_heads, kv_len, head_dim]
        # value_states: [bsz, num_key_value_heads, kv_len, head_dim]

        # Repeat KV if needed
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Q * K^T
        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states_trans = self.qkt_matmul.quant_x2(key_states).transpose(2, 3)
        attn_weights = self.qkt_matmul(query_states, key_states_trans)
        
        attn_weights = attn_weights * self.scaling

        if attention_mask is not None:
             attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Dropout
        if self.training:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        # Attn * V
        attn_weights = self.pv_matmul.quant_x1(attn_weights)
        value_states = self.pv_matmul.quant_x2(value_states)
        attn_output = self.pv_matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        
        attn_output = self.o_proj(attn_output)
        
        self.init_duquant_params = torch.tensor(1)

        return attn_output

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)


class QuantFastdLLMDecoderLayer(nn.Module):
    def __init__(self, 
                 config: Fast_dLLM_QwenConfig,
                 ori_layer,
                 args):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QuantFastdLLMAttention(
            org_module=ori_layer.self_attn,
            config=config,
            args=args,
            )
        self.mlp = QuantFastdLLMMLP(
            org_module=ori_layer.mlp,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            args=args,
        )
        self.input_layernorm = DuLlamaRMSNorm(ori_layer.input_layernorm, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DuLlamaRMSNorm(ori_layer.post_attention_layernorm, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[ori_layer.self_attn.layer_idx] if hasattr(config, "layer_types") else "full_attention"

        # Parameters for DuQuant/SmoothQuant
        self.let = args.let

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        update_past_key_values: Optional[bool] = False,
        use_block_cache: Optional[bool] = False,
        block_past_key_values: Optional[Tuple[torch.Tensor]] = None,
        replace_position: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        
        # print(f"DEBUG: DecoderLayer input type: {type(hidden_states)}")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            cache_position=cache_position,
            update_past_key_values=update_past_key_values,
            block_past_key_values=block_past_key_values,
            replace_position=replace_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        names = []
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                names.append(name)
                m.set_quant_state(weight_quant, act_quant)
      
    def smooth_and_quant_temporary(self):
        if self.let:
            with torch.no_grad():
                for name, module in self.named_parameters():
                    if "smooth_scale" in name:
                        module.data = truncate_number(module)
            smooth_ln_fcs_temporary(self.input_layernorm,[self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            smooth_ln_fcs_temporary(self.post_attention_layernorm,[self.mlp.up_proj,self.mlp.gate_proj],
                                    self.fc1_smooth_scale,self.fc1_smooth_shift)
            smooth_fc_fc_temporary(self.self_attn.v_proj,self.self_attn.o_proj,
                                self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_temporary(self.self_attn.q_proj, self.self_attn.k_proj,
                                self.qkt_smooth_scale)
            self.mlp.down_proj.temp_weight = self.mlp.down_proj.weight
        else:
            for name, module in self.named_modules():
                if isinstance(module, QuantLinear):
                    module.temp_weight = module.weight
        # quant
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                if hasattr(module, "temp_weight"):
                    module.temp_weight = module.weight_quantizer(module.temp_weight)
                else:
                    module.temp_weight = module.weight_quantizer(module.weight)
                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter=True

    def clear_temp_variable(self):
       for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        if self.let:
            for name, module in self.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
            smooth_ln_fcs_inplace(self.input_layernorm,[self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            smooth_ln_fcs_inplace(self.post_attention_layernorm,[self.mlp.up_proj,self.mlp.gate_proj],
                                    self.fc1_smooth_scale,self.fc1_smooth_shift)
            smooth_fc_fc_inplace(self.self_attn.v_proj,self.self_attn.o_proj,
                                self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_inplace(self.self_attn.q_proj, self.self_attn.k_proj,
                                self.qkt_smooth_scale)
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter=False

    def let_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)  

    def lwc_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1:
                params.append(m)
        return iter(params)  

    def duquant_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1 or n.find(template) > -1:
                params.append(m)
        return iter(params)  
    
    def duquant_state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if name.find('smooth') > -1 or name.find('bound_factor') > -1:
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination
    
    def register_scales_and_zeros(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_scales_and_zeros()
    
    def register_duquant_params(self):        
        for name, module in self.named_modules():
            if isinstance(module, QuantFastdLLMMLP) or isinstance(module, QuantFastdLLMAttention):
                delattr(module, 'init_duquant_params')
                module.register_buffer('init_duquant_params', torch.tensor(1))
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_duquant_params()
                module.act_quantizer.register_duquant_params()
    
    def load_duquant_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('R') > -1 or k.find('permutation_list') > -1 or k.find('init_duquant_params') > -1:
                exec(f'self.{k} = v.to(device)')
    
    def load_smooth_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('smooth') > -1:
                # exec(f'self.{k} = v')
                self.register_parameter(k, torch.nn.Parameter(v.to(device), requires_grad=False))
    
    def load_post_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('post') > -1:
                # exec(f'self.{k} = v')
                rg = False if k.find('down') > -1 else True
                self.register_parameter(k, torch.nn.Parameter(v.to(device), requires_grad=rg))

    def load_lwc_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('bound_factor') > -1:
                v = torch.nn.Parameter(v.to(device))
                exec(f'self.{k} = v.to(device)')
