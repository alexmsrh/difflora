'''
Some necessary subclasses of transformers Llama classes for Differential Transformer
Parts of code reused from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
'''

import os
import math
from typing import Optional, Tuple, Union, Callable

import torch
from torch import nn
from safetensors.torch import load_file
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaForCausalLM, LlamaModel, LlamaMLP, LlamaAttention, LlamaFlashAttention2, LlamaRMSNorm, apply_rotary_pos_emb, repeat_kv
from models.generators.DiffTransformer.custom_flash_attn import flash_attn_func
from models.generators.DiffTransformer.llama_lora_diff_transformer_config import LlamaLoraDiffTransformerConfig
from transformers.cache_utils import Cache, StaticCache
from transformers.utils import logging

logger = logging.get_logger(__name__)

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class LlamaLoRAMLP(LlamaMLP, nn.Module):
    def __init__(self, config: LlamaLoraDiffTransformerConfig):
        super().__init__(config)

        # define adapters
        # TODO: pass different r for MLP instead of attention_lora_r
        self.gate_proj_lora_A = nn.Linear(self.hidden_size, config.attention_lora_r, bias=False)
        self.gate_proj_lora_B = nn.Linear(config.attention_lora_r, self.intermediate_size, bias=False)
        self.up_proj_lora_A = nn.Linear(self.hidden_size, config.attention_lora_r, bias=False)
        self.up_proj_lora_B = nn.Linear(config.attention_lora_r, self.intermediate_size, bias=False)
        self.down_proj_lora_A = nn.Linear(self.intermediate_size, config.attention_lora_r, bias=False)
        self.down_proj_lora_B = nn.Linear(config.attention_lora_r, self.hidden_size, bias=False)
        self.lora_dropout = nn.Dropout(p=config.attention_lora_dropout)
        self.lora_scaling = config.attention_lora_alpha / config.attention_lora_r if config.attention_lora_r is not None and config.attention_lora_alpha is not None else 1.0

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            raise NotImplementedError("Pretraining tensor parallel not implemented for LlamaLoRAMLP")
        else:
            # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            gate_proj = self.gate_proj(x) + self.gate_proj_lora_B(self.gate_proj_lora_A(self.lora_dropout(x))) * self.lora_scaling
            up_proj = self.up_proj(x) + self.up_proj_lora_B(self.up_proj_lora_A(self.lora_dropout(x))) * self.lora_scaling
            act = self.act_fn(gate_proj) * up_proj
            down_proj = self.down_proj(act) + self.down_proj_lora_B(self.down_proj_lora_A(self.lora_dropout(act))) * self.lora_scaling

        return down_proj
    
    def init_lora_mlp(self):
        nn.init.kaiming_uniform_(self.gate_proj_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate_proj_lora_B.weight)
        nn.init.kaiming_uniform_(self.up_proj_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj_lora_B.weight)
        nn.init.kaiming_uniform_(self.down_proj_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj_lora_B.weight)

    def freeze_parameters(self):
        """
        Freeze all parameters except for the LoRA parameters
        """
        for name, param in self.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    
class DiffAttentionMixin:
    def init_diff_attn_lora(self):
        """ same init as https://github.com/huggingface/peft/blob/a4f35971cda2bace54b297ad797ebc98a8f50292/src/peft/tuners/lora/layer.py#L158 """

        if not self.lora_negative_term_only:
            nn.init.kaiming_uniform_(self.wq_lora_A1.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.wk_lora_A1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.wq_lora_B1.weight)
            nn.init.zeros_(self.wk_lora_B1.weight)

        if self.negative_term_full_dim:
            nn.init.kaiming_uniform_(self.wq_2.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.wk_2.weight, a=math.sqrt(5))
        elif self.negative_term_lora_only: # if adapters only gradients don't propagate if B is 0
            nn.init.kaiming_uniform_(self.wq_lora_B2.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.wk_lora_B2.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.wq_lora_A2.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.wk_lora_A2.weight, a=math.sqrt(5))
        else:
            nn.init.zeros_(self.wq_lora_B2.weight)
            nn.init.zeros_(self.wk_lora_B2.weight)
            nn.init.kaiming_uniform_(self.wq_lora_A2.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.wk_lora_A2.weight, a=math.sqrt(5))

        if self.config.lora_v:
            nn.init.kaiming_uniform_(self.wv_lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.wv_lora_B.weight)
        if self.config.lora_o:
            nn.init.kaiming_uniform_(self.wo_lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.wo_lora_B.weight)

    def reset_weights_from_base_model(self):
        """
        Reset the weights W_Q, W_K, W_V, W_O
        """
        nn.init.kaiming_uniform_(self.q_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.k_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.o_proj.weight, a=math.sqrt(5))

    def set_weights_from_base_model(self, base_model_layer_attn):
        """
        Set the weights W_Q, W_K, W_V, W_O to the base model weights
        """
        self.q_proj.weight.data = base_model_layer_attn.q_proj.weight.data.clone()
        self.k_proj.weight.data = base_model_layer_attn.k_proj.weight.data.clone()
        self.v_proj.weight.data = base_model_layer_attn.v_proj.weight.data.clone()
        self.o_proj.weight.data = base_model_layer_attn.o_proj.weight.data.clone()

    def freeze_parameters(self):
        """
        Freeze all parameters except for the LoRA parameters
        """
        for name, param in self.named_parameters():
            if 'lambda' in name or 'lora' in name or 'subln' in name or (self.negative_term_full_dim and ('wq_2' in name or 'wk_2' in name)):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def extra_repr(self):
        """
        overloads the nn.Module method to include lambdas and/or other diff-attn stuff when printing model 
        (some stuff is not printed by default because they are not named submodules, just parameters)
        """
        lambdas_repr = ""
        if self.learn_lambda:
            lambdas_repr = f"(lambda_q1): Parameter({self.lambda_q1.shape})\n(lambda_k1): Parameter({self.lambda_k1.shape})\n(lambda_q2): Parameter({self.lambda_q2.shape})\n(lambda_k2): Parameter({self.lambda_k2.shape})"
        else:
            lambdas_repr = f"(lambda_fixed): {self.lambda_init}"
        if self.relu:
            lambdas_repr += f"\n(relu_on_differential): {self.relu}"
        return  lambdas_repr


class LlamaLoraDiffAttention(DiffAttentionMixin, LlamaAttention):
    """Multi-headed differential attention from 'Differential Transformer' paper: https://arxiv.org/abs/2410.05258"""

    def __init__(self, config: LlamaLoraDiffTransformerConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.learn_lambda = config.learn_lambda
        if self.learn_lambda:
            self.lambda_init = config.diff_attn_lambda if config.diff_attn_lambda != 0 else lambda_init_fn(layer_idx)
            self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1), requires_grad=True)
            self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1), requires_grad=True)
            self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1), requires_grad=True)
            self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1), requires_grad=True)
            #self.lambda_dropout = nn.Dropout(p=config.lambda_dropout)

        else:
            #self.lambda_init = config.diff_attn_lambda if isinstance(config.diff_attn_lambda, float) else config.diff_attn_lambda[layer_idx]
            self.lambda_init = config.diff_attn_lambda if isinstance(config.diff_attn_lambda, float) else config.diff_attn_lambda[layer_idx]
        
        self.lora_negative_term_only = config.lora_negative_term_only
        self.negative_term_lora_only = config.negative_term_lora_only
        self.negative_term_full_dim = config.negative_term_full_dim
        if not self.lora_negative_term_only:
            self.wq_lora_A1 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wq_lora_B1 = nn.Linear(config.attention_lora_r, self.num_heads * self.head_dim, bias=False)
            self.wk_lora_A1 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wk_lora_B1 = nn.Linear(config.attention_lora_r, self.num_key_value_heads * self.head_dim, bias=False)

        if self.negative_term_full_dim:
            self.wq_2 = nn.Linear(self.num_heads * self.head_dim, self.num_heads * self.head_dim, bias=False)
            self.wk_2 = nn.Linear(self.num_heads * self.head_dim, self.num_key_value_heads * self.head_dim, bias=False)
        else:
            self.wq_lora_A2 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wq_lora_B2 = nn.Linear(config.attention_lora_r, self.num_heads * self.head_dim, bias=False)
            self.wk_lora_A2 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wk_lora_B2 = nn.Linear(config.attention_lora_r, self.num_key_value_heads * self.head_dim, bias=False)

        if config.lora_v:
            self.wv_lora_A = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wv_lora_B = nn.Linear(config.attention_lora_r, self.num_key_value_heads * self.head_dim, bias=False)
        if config.lora_o:
            self.wo_lora_A = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wo_lora_B = nn.Linear(config.attention_lora_r, self.num_heads * self.head_dim, bias=False)

        self.lora_dropout = nn.Dropout(p=config.attention_lora_dropout)
        self.lora_scaling = config.attention_lora_alpha / config.attention_lora_r if config.attention_lora_r is not None and config.attention_lora_alpha is not None else 1.0
        self.subln = LlamaRMSNorm(self.head_dim, eps=1e-5) if config.groupnorm else None
        self.relu = nn.ReLU() if config.relu_on_differential else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size() # X = (batch, q_len, hidden_dim) where hidden_dim = num_heads * head_dim; note in QA task q_len > 1 in first pass and q_len=1 in next passes (cached context);
        if self.config.pretraining_tp > 1:
            raise NotImplementedError("Pretraining tensor parallel not implemented for LlamaDiffAttention")

        else:
            query_states = self.q_proj(hidden_states) # X @ W_q = (b, q_len, hidden_dim) @ (hidden_dim, self.num_heads * self.head_dim) = (b, q_len, self.num_heads * self.head_dim)  where self.num_heads * self.head_dim = hidden_dim = 32*128 = 4096
            key_states = self.k_proj(hidden_states)   # X @ W_k = (b, q_len, hidden_dim) @ (hidden_dim, self.num_key_value_heads * self.head_dim) = (b, q_len, self.num_key_value_heads * self.head_dim)
            value_states = self.v_proj(hidden_states) # X @ W_v = (b, q_len, hidden_dim) @ (hidden_dim, self.num_key_value_heads * self.head_dim) = (b, q_len, self.num_key_value_heads * self.head_dim)
            # assert all((
            #     query_states.size() == torch.Size([bsz, q_len, self.num_heads * self.head_dim]),
            #     key_states.size() == torch.Size([bsz, q_len, self.num_key_value_heads * self.head_dim]),
            #     value_states.size() == torch.Size([bsz, q_len, self.num_key_value_heads * self.head_dim]),
            # )), f"query_states.size() = {query_states.size()}, key_states.size() = {key_states.size()}, value_states.size() = {value_states.size()}"

            # compute adapter query/key states
            if not self.lora_negative_term_only:
                lora_query_states_1 = self.wq_lora_B1(self.wq_lora_A1(self.lora_dropout(hidden_states))) * self.lora_scaling # X @ wq_A1 @ wq_B1 = (b, q_len, hidden_dim) @ (hidden_dim, r) @ (r, hidden_dim) = (b, q_len, hidden_dim)
                lora_key_states_1 = self.wk_lora_B1(self.wk_lora_A1(self.lora_dropout(hidden_states))) * self.lora_scaling # X @ wk_A1 @ wk_B1 = (b, q_len, hidden_dim) @ (hidden_dim, r) @ (r, num_key_value_heads * head_dim) = (b, q_len, self.num_key_value_heads * self.head_dim)
            if self.negative_term_full_dim:
                lora_query_states_2 = self.wq_2(self.lora_dropout(hidden_states))
                lora_key_states_2 = self.wk_2(self.lora_dropout(hidden_states))
            else:
                lora_query_states_2 = self.wq_lora_B2(self.wq_lora_A2(self.lora_dropout(hidden_states))) * self.lora_scaling # same as query_states_1
                lora_key_states_2 = self.wk_lora_B2(self.wk_lora_A2(self.lora_dropout(hidden_states))) * self.lora_scaling # same as key_states_1
            # if not self.lora_negative_term_only:
            #     assert all((
            #         lora_query_states_1.size() == torch.Size([bsz, q_len, self.num_heads * self.head_dim]),
            #         lora_query_states_2.size() == torch.Size([bsz, q_len, self.num_heads * self.head_dim]),
            #         lora_key_states_1.size() == torch.Size([bsz, q_len, self.num_key_value_heads * self.head_dim]),
            #         lora_key_states_2.size() == torch.Size([bsz, q_len, self.num_key_value_heads * self.head_dim]),
            #     )), f"lora_query_states_1.size() = {lora_query_states_1.size()}, lora_query_states_2.size() = {lora_query_states_2.size()}, lora_key_states_1.size() = {lora_key_states_1.size()}, lora_key_states_2.size() = {lora_key_states_2.size()}"

            if self.config.lora_v:
                lora_value_states = self.wv_lora_B(self.wv_lora_A(self.lora_dropout(hidden_states))) * self.lora_scaling
            if self.config.lora_o:
                lora_output_states = self.wo_lora_B(self.wo_lora_A(self.lora_dropout(hidden_states))) * self.lora_scaling

            # add adapter query/key states to original query/key states
            if not self.lora_negative_term_only:
                query_states_1 = query_states + lora_query_states_1 # (b, q_len, hidden_dim)
                key_states_1 = key_states + lora_key_states_1 # (b, q_len, self.num_key_value_heads * self.head_dim)
            else:
                query_states_1 = query_states
                key_states_1 = key_states
            if self.negative_term_lora_only or self.negative_term_full_dim:
                query_states_2 = lora_query_states_2
                key_states_2 = lora_key_states_2
            else:
                query_states_2 = query_states + lora_query_states_2 # (b, q_len, hidden_dim)
                key_states_2 = key_states + lora_key_states_2 # (b, q_len, self.num_key_value_heads * self.head_dim)
            if self.config.lora_v:
                value_states = value_states + lora_value_states

        query_states_1 = query_states_1.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_heads, q_len, head_dim)
        query_states_2 = query_states_2.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_heads, q_len, head_dim)
        key_states_1 = key_states_1.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_key_value_heads, q_len, head_dim)
        key_states_2 = key_states_2.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_key_value_heads, q_len, head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_key_value_heads, q_len, head_dim)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states_1, key_states_1 = apply_rotary_pos_emb(query_states_1, key_states_1, cos, sin)
        query_states_2, key_states_2 = apply_rotary_pos_emb(query_states_2, key_states_2, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

            # concatenate key_states along the head_dim dimension for cache 
            # this way we can store both key_1 and key_2 in the cache without modifying how cache works
            key_states_cache = torch.cat([key_states_1, key_states_2], dim=-1) # concat to shape (b, num_key_value_heads, q_len, 2 * head_dim)
            key_states_cache, value_states = past_key_value.update(key_states_cache, value_states, self.layer_idx, cache_kwargs)
            total_q_len = key_states_cache.size(-2)
            key_states_1, key_states_2 = key_states_cache.split(self.head_dim, dim=-1) # split each key back to shape (b, num_key_value_heads, q_len, head_dim)
            # assert all((
            #     key_states_1.size() == torch.Size([bsz, self.num_key_value_heads, total_q_len, self.head_dim]),
            #     key_states_2.size() == torch.Size([bsz, self.num_key_value_heads, total_q_len, self.head_dim]),
            #     value_states.size() == torch.Size([bsz, self.num_key_value_heads, total_q_len, self.head_dim]),
            # )), f"key_states_1.size() = {key_states_1.size()}, key_states_2.size() = {key_states_2.size()}, value_states.size() = {value_states.size()}"

        total_q_len = key_states_1.size(-2)
        key_states_1 = repeat_kv(key_states_1, self.num_key_value_groups) # (b, num_key_value_heads, q_len, head_dim) -> (b, num_heads, q_len, head_dim)
        key_states_2 = repeat_kv(key_states_2, self.num_key_value_groups) # same
        value_states = repeat_kv(value_states, self.num_key_value_groups) # (b, num_key_value_heads, q_len, head_dim) -> (b, num_heads, q_len, head_dim)
        # assert all((
        #     query_states_1.size() == torch.Size([bsz, self.num_heads, q_len, self.head_dim]),
        #     query_states_2.size() == torch.Size([bsz, self.num_heads, q_len, self.head_dim]),
        #     key_states_1.size() == torch.Size([bsz, self.num_heads, total_q_len, self.head_dim]),
        #     key_states_2.size() == torch.Size([bsz, self.num_heads, total_q_len, self.head_dim]),
        #     value_states.size() == torch.Size([bsz, self.num_heads, total_q_len, self.head_dim]),
        # )), f"query_states_1.size() = {query_states_1.size()}, query_states_2.size() = {query_states_2.size()}, key_states_1.size() = {key_states_1.size()}, key_states_2.size() = {key_states_2.size()}, value_states.size() = {value_states.size()}"

        attn_weights_1 = torch.matmul(query_states_1, key_states_1.transpose(2, 3)) / math.sqrt(self.head_dim) # (b, num_heads, q_len, head_dim) @ (b, num_heads, q_len, head_dim).T(2,3) -> (b, num_heads, q_len, q_len)
        attn_weights_2 = torch.matmul(query_states_2, key_states_2.transpose(2, 3)) / math.sqrt(self.head_dim) # same
        # assert all((
        #     attn_weights_1.size() == torch.Size([bsz, self.num_heads, q_len, total_q_len]),
        #     attn_weights_2.size() == torch.Size([bsz, self.num_heads, q_len, total_q_len]),
        # )), f"attn_weights_1.size() = {attn_weights_1.size()}, attn_weights_2.size() = {attn_weights_2.size()}"

        if attention_mask is not None:  # no matter the length, we just slice it
            # TODO: this can if loop probably be removed since attn_implementation bug is fixed
            # if attention_mask.dim() == 2: # depending on gpu type and inference setup (training or not, flash-attention, etc) attention mask can be 2D or 4D
            #     # Expand attention mask to 4D
            #     attention_mask = attention_mask[:, None, None, :].expand(-1, 1, hidden_states.size(1), -1) # TODO: check this is correct (+ implement flash attention)
            causal_mask = attention_mask[:, :, :, : key_states_1.shape[-2]] # (b, 1, q_len, q_len)
            
            attn_weights_1 = attn_weights_1 + causal_mask
            attn_weights_2 = attn_weights_2 + causal_mask

        # upcast attention to fp32
        attn_weights_1 = nn.functional.softmax(attn_weights_1, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_2 = nn.functional.softmax(attn_weights_2, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_1 = nn.functional.dropout(attn_weights_1, p=self.attention_dropout, training=self.training)
        attn_weights_2 = nn.functional.dropout(attn_weights_2, p=self.attention_dropout, training=self.training)
        if self.learn_lambda:
            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(query_states)
            lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(query_states)
            
            # lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(query_states)
            # lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(query_states)
            lambda_full = lambda_1 - lambda_2 + self.lambda_init
        else:
            lambda_full = self.lambda_init

        attn_weights = attn_weights_1 - lambda_full * attn_weights_2 # diff attn
        if self.relu:
            attn_weights = self.relu(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states) # (b, num_heads, q_len, q_len) @ (b, num_heads, q_len, head_dim) -> (b, num_heads, q_len, head_dim)
        # assert attn_output.size() == torch.Size([bsz, self.num_heads, q_len, self.head_dim]), f"attn_output.size() = {attn_output.size()}"
        # GroupNorm is layer normalization but applied to each head independently
        if self.subln is not None:
            attn_output = self.subln(attn_output)
            attn_output = attn_output * (1 - self.lambda_init)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous() # (b, num_heads, q_len, head_dim) -> (b, q_len, num_heads, head_dim)

        attn_output = attn_output.reshape(bsz, q_len, -1) # (b, q_len, num_heads, head_dim) -> (b, q_len, num_heads * head_dim) = (b, q_len, hidden_dim)

        attn_output = self.o_proj(attn_output) # (b, q_len, hidden_dim) @ (hidden_dim, hidden_dim) = (b, q_len, hidden_dim)
        if self.config.lora_o:
            attn_output = attn_output + lora_output_states

        if not output_attentions:
            attn_weights = None
        elif output_attentions:
            attn_weights = torch.cat([attn_weights.detach().clone().unsqueeze(2), attn_weights_1.detach().clone().unsqueeze(2), lambda_full * attn_weights_2.detach().clone().unsqueeze(2)], dim=2) # (b, num_heads, q_len, q_len) -> (b, num_heads, 2, q_len, q_len)

        return attn_output, attn_weights, past_key_value
    
    def extra_repr(self):
        return super().extra_repr()

class LlamaLoraFlashDiffAttention2(DiffAttentionMixin, LlamaFlashAttention2):
    """Flash attention implementation using https://github.com/xiayuqing0622/flex_head_fa """

    def __init__(self, config: LlamaLoraDiffTransformerConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.learn_lambda = config.learn_lambda
        if self.learn_lambda:
            self.lambda_init = config.diff_attn_lambda if config.diff_attn_lambda != 0 else lambda_init_fn(layer_idx)
            self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float16).normal_(mean=0,std=0.1), requires_grad=True)
            self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float16).normal_(mean=0,std=0.1), requires_grad=True)
            self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float16).normal_(mean=0,std=0.1), requires_grad=True)
            self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float16).normal_(mean=0,std=0.1), requires_grad=True)
        else:
            self.lambda_init = config.diff_attn_lambda if isinstance(config.diff_attn_lambda, float) else config.diff_attn_lambda[layer_idx]

        self.lora_negative_term_only = config.lora_negative_term_only
        self.negative_term_lora_only = config.negative_term_lora_only
        self.negative_term_full_dim = config.negative_term_full_dim
        if not self.lora_negative_term_only:
            self.wq_lora_A1 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wq_lora_B1 = nn.Linear(config.attention_lora_r, self.num_heads * self.head_dim, bias=False)
            self.wk_lora_A1 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wk_lora_B1 = nn.Linear(config.attention_lora_r, self.num_key_value_heads * self.head_dim, bias=False)

        if self.negative_term_full_dim:
            self.wq_2 = nn.Linear(self.num_heads * self.head_dim, self.num_heads * self.head_dim, bias=False)
            self.wk_2 = nn.Linear(self.num_heads * self.head_dim, self.num_key_value_heads * self.head_dim, bias=False)
        else:
            self.wq_lora_A2 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wq_lora_B2 = nn.Linear(config.attention_lora_r, self.num_heads * self.head_dim, bias=False)
            self.wk_lora_A2 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wk_lora_B2 = nn.Linear(config.attention_lora_r, self.num_key_value_heads * self.head_dim, bias=False)
        
        if config.lora_v:
            self.wv_lora_A = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wv_lora_B = nn.Linear(config.attention_lora_r, self.num_key_value_heads * self.head_dim, bias=False)
        if config.lora_o:
            self.wo_lora_A = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wo_lora_B = nn.Linear(config.attention_lora_r, self.num_heads * self.head_dim, bias=False)

        self.lora_dropout = nn.Dropout(p=config.attention_lora_dropout)
        self.lora_scaling = config.attention_lora_alpha / config.attention_lora_r if config.attention_lora_r is not None and config.attention_lora_alpha is not None else 1.0
        self.subln = LlamaRMSNorm(self.head_dim, eps=1e-5) if config.groupnorm else None
        self.deterministic_backward = config.fa_deterministic_backward
        self.relu = nn.ReLU() if config.relu_on_differential else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False
        bsz, q_len, _ = hidden_states.size()  # X = (batch, q_len, hidden_dim) where hidden_dim = num_heads * head_dim; note in QA task q_len > 1 in first pass and q_len=1 in next passes (cached context);

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if not self.lora_negative_term_only:
            lora_query_states_1 = self.wq_lora_B1(self.wq_lora_A1(self.lora_dropout(hidden_states))) * self.lora_scaling # X @ wq_A1 @ wq_B1 = (b, q_len, hidden_dim) @ (hidden_dim, r) @ (r, hidden_dim) = (b, q_len, hidden_dim)
            lora_key_states_1 = self.wk_lora_B1(self.wk_lora_A1(self.lora_dropout(hidden_states))) * self.lora_scaling # X @ wk_A1 @ wk_B1 = (b, q_len, hidden_dim) @ (hidden_dim, r) @ (r, num_key_value_heads * head_dim) = (b, q_len, self.num_key_value_heads * self.head_dim)
        if self.negative_term_full_dim:
            lora_query_states_2 = self.wq_2(self.lora_dropout(hidden_states))
            lora_key_states_2 = self.wk_2(self.lora_dropout(hidden_states))
        else:
            lora_query_states_2 = self.wq_lora_B2(self.wq_lora_A2(self.lora_dropout(hidden_states))) * self.lora_scaling # same as query_states_1
            lora_key_states_2 = self.wk_lora_B2(self.wk_lora_A2(self.lora_dropout(hidden_states))) * self.lora_scaling # same as key_states_1

        if self.config.lora_v:
            lora_value_states = self.wv_lora_B(self.wv_lora_A(self.lora_dropout(hidden_states))) * self.lora_scaling
        if self.config.lora_o:
            lora_output_states = self.wo_lora_B(self.wo_lora_A(self.lora_dropout(hidden_states))) * self.lora_scaling

        if not self.lora_negative_term_only:
            query_states_1 = query_states + lora_query_states_1 # (b, q_len, hidden_dim)
            key_states_1 = key_states + lora_key_states_1 # (b, q_len, self.num_key_value_heads * self.head_dim)
        else:
            query_states_1 = query_states
            key_states_1 = key_states
        if self.negative_term_lora_only or self.negative_term_full_dim:
            query_states_2 = lora_query_states_2
            key_states_2 = lora_key_states_2
        else:
            query_states_2 = query_states + lora_query_states_2 # (b, q_len, hidden_dim)
            key_states_2 = key_states + lora_key_states_2 # (b, q_len, self.num_key_value_heads * self.head_dim)
        if self.config.lora_v:
            value_states = value_states + lora_value_states

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states_1 = query_states_1.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_heads, q_len, head_dim)
        query_states_2 = query_states_2.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_heads, q_len, head_dim)
        key_states_1 = key_states_1.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_key_value_heads, q_len, head_dim)
        key_states_2 = key_states_2.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_key_value_heads, q_len, head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_key_value_heads, q_len, head_dim)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states_1, key_states_1 = apply_rotary_pos_emb(query_states_1, key_states_1, cos, sin)
        query_states_2, key_states_2 = apply_rotary_pos_emb(query_states_2, key_states_2, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

            key_states_cache = torch.cat([key_states_1, key_states_2], dim=-1) # concat to shape (b, num_key_value_heads, q_len, 2 * head_dim)
            key_states_cache, value_states = past_key_value.update(key_states_cache, value_states, self.layer_idx, cache_kwargs)
            # total_q_len = key_states_cache.size(-2)
            key_states_1, key_states_2 = key_states_cache.split(self.head_dim, dim=-1) # split each key back to shape (b, num_key_value_heads, q_len, head_dim)
            # assert all((
            #     key_states_1.size() == torch.Size([bsz, self.num_key_value_heads, total_q_len, self.head_dim]),
            #     key_states_2.size() == torch.Size([bsz, self.num_key_value_heads, total_q_len, self.head_dim]),
            #     value_states.size() == torch.Size([bsz, self.num_key_value_heads, total_q_len, self.head_dim]),
            # )), f"key_states_1.size() = {key_states_1.size()}, key_states_2.size() = {key_states_2.size()}, value_states.size() = {value_states.size()}"

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states_1 = query_states_1.transpose(1, 2)
        query_states_2 = query_states_2.transpose(1, 2)
        key_states_1 = key_states_1.transpose(1, 2)
        key_states_2 = key_states_2.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            # query_states = query_states.to(target_dtype)
            # key_states = key_states.to(target_dtype)
            query_states_1 = query_states_1.to(target_dtype)
            query_states_2 = query_states_2.to(target_dtype)
            key_states_1 = key_states_1.to(target_dtype)
            key_states_2 = key_states_2.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output_1 = flash_attn_func(query_states_1, key_states_1, value_states, dropout_p=dropout_rate, causal=True, deterministic=self.deterministic_backward)
        attn_output_2 = flash_attn_func(query_states_2, key_states_2, value_states, dropout_p=dropout_rate, causal=True, deterministic=self.deterministic_backward)

        if self.learn_lambda:
            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(query_states)
            lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(query_states)
            lambda_full = lambda_1 - lambda_2 + self.lambda_init
        else:
            lambda_full = self.lambda_init

        # Differential Attn: A = (sm(Q1K1^T/sqrt(d)) - lambda * sm(Q2K2^T/sqrt(d))) @ V
        #              = sm(Q1K1^T/sqrt(d) @ V - lambda * sm(Q2K2^T/sqrt(d)) @ V
        #              = flashattention(Q1, K1, V) - lambda * flashattention(Q2, K2, V)
        attn_output = attn_output_1 - lambda_full * attn_output_2

        if self.subln is not None:
            attn_output = self.subln(attn_output)
            attn_output = attn_output * (1 - self.lambda_init)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if self.config.lora_o:
            attn_output = attn_output + lora_output_states

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def extra_repr(self):
        return super().extra_repr()
    

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaLoraDiffAttention,
    "flash_attention_2": LlamaLoraFlashDiffAttention2,
}

class LlamaLoraDiffTransformerModel(LlamaModel):
    config_class = LlamaLoraDiffTransformerConfig

    def __init__(self, config: LlamaLoraDiffTransformerConfig):
        super().__init__(config)
        for layer_idx,layer in enumerate(self.layers):
            if isinstance(layer.self_attn, LlamaAttention) and layer_idx in config.layers_to_transform:
                # HF might set config._attn_implementation to 'sdpa' by default when loading a checkpoint so we need a custom attribute for diff attn implementation
                if hasattr(config, "diff_attn_implementation"):
                    attn_implementation = config.diff_attn_implementation
                elif hasattr(config, "_attn_implementation"):
                    attn_implementation = config._attn_implementation
                else:
                    print(f"WARNING: [loading attn at layer {layer_idx}] no attn implementation found in config. Setting it to 'eager'.")
                    attn_implementation = "eager"
                if attn_implementation not in LLAMA_ATTENTION_CLASSES:
                    print(f"WARNING: [loading attn at layer {layer_idx}] attn implementation `{attn_implementation}` is unknown or not implemented for diff attention. Setting it to 'eager'.")
                    attn_implementation = "eager"
                layer.self_attn = LLAMA_ATTENTION_CLASSES[attn_implementation](config=config, layer_idx=layer_idx)
            if config.lora_mlp and layer_idx in config.layers_to_transform:
                layer.mlp = LlamaLoRAMLP(config)

class LlamaLoraDiffTransformerForCausalLM(LlamaForCausalLM, GenerationMixin):
    # edit base __init__ to change self.model + freeze params + load base model weights + other configs if any
    def __init__(self, config: LlamaLoraDiffTransformerConfig, base_model: LlamaForCausalLM = None):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaLoraDiffTransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        for layer in self.model.layers:
            if isinstance(layer.self_attn, (LlamaLoraDiffAttention, LlamaLoraFlashDiffAttention2)):
                layer.self_attn.init_diff_attn_lora()
            if isinstance(layer.mlp, LlamaLoRAMLP):
                layer.mlp.init_lora_mlp()
        print(f"Initialized diff transformer weights for layer(s) {set(list(range(config.num_hidden_layers))).intersection(set(config.layers_to_transform))}")

        # freeze all params (except attention)
        for _, param in self.named_parameters():
            param.requires_grad = False
        for i,layer in enumerate(self.model.layers):
            if i in config.layers_to_transform:
                layer.self_attn.freeze_parameters() # this activates the LoRA parameters
                if config.lora_mlp:
                    layer.mlp.freeze_parameters()
            else:
                for _, param in layer.named_parameters():
                    param.requires_grad = False

        # load the state dict of the base model everywhere (except the custom parameters)
        if base_model:
            self.load_base_weights(base_model.state_dict())
            
        # optionally reset all attn weights
        if not config.diff_attn_init_with_base_weights:
            print("Resetting attn weights for layer(s) ", config.layers_to_transform)
            for i,layer in enumerate(self.model.layers):
                if i in config.layers_to_transform:
                    layer.self_attn.reset_weights_from_base_model()
        self.enable_input_require_grads()  # needed for gradient checkpointing: https://github.com/huggingface/peft/issues/137

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        # save only trainable params
        state_dict = self.state_dict()
        for name, param in self.named_parameters():
            if not param.requires_grad:
                del state_dict[name]
        super().save_pretrained(save_directory, is_main_process, state_dict, save_function, push_to_hub, max_shard_size, safe_serialization, variant, token, save_peft_format, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        weights_only: bool = True,
        **kwargs,
    ) -> "PreTrainedModel":
        """
        Load base model and then load the custom model on top of it
        """
        config = LlamaLoraDiffTransformerConfig.from_pretrained(pretrained_model_name_or_path)
        if "attn_implementation" in kwargs and kwargs["attn_implementation"] is not None and kwargs["attn_implementation"] != config.diff_attn_implementation:
            print(f"WARNING: you are loading a model with attention implementation = {kwargs['attn_implementation']} which is different than the one in the model config = {config.diff_attn_implementation}.")
            config.diff_attn_implementation = kwargs["attn_implementation"]

        base_model_path = config._name_or_path # something like meta-llama/Meta-Llama-3-8B-Instruct
        print("=============== YOU CAN SAFELY IGNORE THE MISSING KEYS WARNING BELOW ===============")
        model = super().from_pretrained(base_model_path, *model_args, config=config, cache_dir=cache_dir, ignore_mismatched_sizes=ignore_mismatched_sizes, force_download=force_download, local_files_only=local_files_only, token=token, revision=revision, use_safetensors=use_safetensors, weights_only=weights_only, **kwargs)
        print("=============== YOU CAN SAFELY IGNORE THE MISSING KEYS WARNING ABOVE ===============")

        # Now we load the adapters. Load all the model.safetensors files TODO: cleaner with cases if multiple shards
        if os.path.exists(f"{pretrained_model_name_or_path}/model.safetensors"):
            adapters_state_dict = load_file(f"{pretrained_model_name_or_path}/model.safetensors")
        elif os.path.exists(f"{pretrained_model_name_or_path}/pytorch_model.bin"):
            adapters_state_dict = torch.load(f"{pretrained_model_name_or_path}/pytorch_model.bin")
        else:
            assert False, 'Cannot load adapters...'
        model.load_diff_attn_weights(adapters_state_dict)
        return model
    
    def load_base_weights(self, base_model_state_dict):
        missing_keys, unexpected_keys = self.load_state_dict(base_model_state_dict, strict=False) # TODO: maybe a for loop to load modules 1 by 1 and free memory so we don't store 2 models at the same time
        assert len(unexpected_keys) == 0, "Unexpected keys found in the model state dict. Please check the model architecture."
        if self.config.verbose:
            print("Loaded base weights.")
        missing_keys_without_lora_params = [key for key in missing_keys if 'lora' not in key and 'subln' not in key and 'lambda' not in key and 'wk_2' not in key and 'wq_2' not in key]
        assert len(missing_keys_without_lora_params) == 0, f"Missing keys (excluding LoRA, subln, lambda): {missing_keys_without_lora_params}"

    def load_diff_attn_weights(self, adapters_state_dict):
        missing_keys, unexpected_keys = self.load_state_dict(adapters_state_dict, strict=False)
        if self.config.verbose:
            print("Loaded diff transformer weights.")
            # print("Num missing keys when loading adapters =", len(missing_keys))
            # print("Num keys that are not LoRA, subln, lambda = ", len([key for key in self.state_dict().keys() if 'lora' not in key and 'subln' not in key and 'lambda' not in key]))
        assert len(unexpected_keys) == 0, f"{len(unexpected_keys)} unexpected keys found in the model state dict: \n{unexpected_keys}"
        assert len([key for key in missing_keys if 'lora' in key or 'subln' in key or 'lambda' in key or 'wq_2' in key or 'wk_2' in key]) == 0, f"Missing keys = {missing_keys}"

    def load_base_weights_and_adapters(self, concat_state_dict):
        missing_keys, unexpected_keys = self.load_state_dict(concat_state_dict, strict=True)
        assert len(missing_keys) == 0 and len(unexpected_keys) == 0, f"Missing keys = {len(missing_keys)} || Unexpected keys = {len(unexpected_keys)}"

    def unfreeze_adapters(self):
        for name, param in self.named_parameters():
            if 'lora' in name or 'subln' in name or 'lambda' in name:
                param.requires_grad = True

LlamaLoraDiffTransformerForCausalLM.register_for_auto_class("AutoModelForCausalLM")