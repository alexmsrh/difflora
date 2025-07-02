from typing import List

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class LlamaLoraDiffTransformerConfig(LlamaConfig):
    model_type = "llama_lora_diff_transformer"

    def __init__(self, 
                 diff_attn_implementation: str = "eager",
                 learn_lambda: bool = False,
                 diff_attn_lambda: float = 0.0,
                 layers_to_transform: List[int] = list(range(0,32)),
                 diff_attn_init_with_base_weights: bool = True,
                 lora_negative_term_only: bool = False,
                 negative_term_lora_only: bool = False,
                 negative_term_full_dim: bool = False,
                 attention_lora_alpha: int = 64,
                 attention_lora_r: int = 32,
                 attention_lora_dropout: float = 0.1,
                 lora_v: bool = False,
                 lora_o: bool = False,
                 lora_mlp: bool = False,
                 groupnorm: bool = True,
                 relu_on_differential: bool = False,
                 flash_attn_deterministic_backward: bool = False,
                 verbose: bool = False,
                 **kwargs):
        """
        Args:
        - diff_attn_implementation: The attention implementation to use. Can be "eager" or "flash_attention_2".
        - learn_lambda: Whether to learn the lambda parameter for the Diff Attn loss. Takes precedence over diff_attn_lambda.
        - diff_attn_lambda: The fixed lambda parameter for the Diff Attn. Ignored if learn_lambda is True.
        - layers_to_transform: List of layer indices to apply Diff Attn to.
        - diff_attn_init_with_base_weights: Whether to initialize Diff Attn weights with the base (pre-trained model) weights (for q_proj, k_proj, v_proj, and o_proj)
        - lora_negative_term_only: Whether to only apply adapters on the right/negative term of diff attn.
        - negative_term_lora_only: Whether to apply the adapter only (vs the adapter on top of pre-trained weights) on the negative term of diff attn.
        - negative_term_full_dim: Whether to learn adapters or a full dimensional denoiser (will equal the positive term dimension).
        - attention_lora_alpha: The alpha parameter for the LORA diff attention.
        - attention_lora_r: The rank for the adapters of diff attn.
        - attention_lora_dropout: The dropout rate for the LORA diff attention.
        - lora_v: Whether to put an adapter on V in the attention.
        - lora_o: Whether to put an adapter on O in the attention.
        - lora_mlp: Whether to put adapters on mlp layers.
        - groupnorm: Whether to use GroupNorm (normalization across attention heads) (see diff attn paper).
        - relu_on_differential: Whether to apply ReLU on the differential term i.e. ReLU(softmax(Q1K1)-softmax(Q2K2))
        - flash_attn_deterministic_backward: Whether to use deterministic backward pass for Flash Attention.
        - verbose: Whether to print verbose logs.
        """
        if learn_lambda and diff_attn_lambda > 0.0:
            logger.warning("learn_lambda is True, but diff_attn_lambda is non-zero. Diff Attn lambdas will be learnable.")
        if negative_term_lora_only and negative_term_full_dim:
            raise ValueError("negative_term_lora_only and negative_term_full_dim cannot be True at the same time.")
        if diff_attn_implementation == 'flash_attention_2' and relu_on_differential:
            raise ValueError("ReLU on differential term is not supported with flash_attention_2.")
        if flash_attn_deterministic_backward and diff_attn_implementation != 'flash_attention_2':
            logger.warning("flash_attn_deterministic_backward is True, but attn_implementation is not flash_attention_2. Ignoring.")
        self.diff_attn_implementation = diff_attn_implementation
        self.learn_lambda = learn_lambda
        if isinstance(diff_attn_lambda, float):
            self.diff_attn_lambda = diff_attn_lambda
        else: # is specified for each layer
            self.diff_attn_lambda = list(diff_attn_lambda)
        self.layers_to_transform = list(layers_to_transform)
        self.diff_attn_init_with_base_weights = diff_attn_init_with_base_weights
        self.lora_negative_term_only = lora_negative_term_only
        self.negative_term_lora_only = negative_term_lora_only
        self.negative_term_full_dim = negative_term_full_dim
        self.attention_lora_alpha = attention_lora_alpha
        self.attention_lora_r = attention_lora_r
        self.attention_lora_dropout = attention_lora_dropout
        self.lora_v = lora_v
        self.lora_o = lora_o
        self.lora_mlp = lora_mlp
        self.groupnorm = groupnorm
        self.relu_on_differential = relu_on_differential
        self.fa_deterministic_backward = flash_attn_deterministic_backward
        self.verbose = verbose
        if "attn_implementation" in kwargs and self.relu_on_differential and kwargs["attn_implementation"] == "flash_attention_2":
            logger.warning("ReLU on differential term is not supported with flash_attention_2. Setting relu_on_differential to False.")
            self.relu_on_differential = False
        if "attn_implementation" not in kwargs:
            kwargs["attn_implementation"] = self.diff_attn_implementation
        super().__init__(**kwargs)
        if isinstance(self.diff_attn_lambda, list):
            assert len(self.diff_attn_lambda) == self.num_hidden_layers, "diff_attn_lambda must be a float or a list of floats with length equal to the number of layers. Found length {} and num_hidden_layers {}.".format(len(self.diff_attn_lambda), self.num_hidden_layers)

LlamaLoraDiffTransformerConfig.register_for_auto_class()