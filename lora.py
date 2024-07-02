# Third Party
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("ibm-granite/granite-3b-code-base")

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config=config)

print(model)

# LlamaForCausalLM(
#   (model): LlamaModel(
#     (embed_tokens): Embedding(49152, 2560, padding_idx=0)
#     (layers): ModuleList(
#       (0-31): 32 x LlamaDecoderLayer(
#         (self_attn): LlamaSdpaAttention(
#           (q_proj): Linear(in_features=2560, out_features=2560, bias=True)
#           (k_proj): Linear(in_features=2560, out_features=2560, bias=True)
#           (v_proj): Linear(in_features=2560, out_features=2560, bias=True)
#           (o_proj): Linear(in_features=2560, out_features=2560, bias=True)
#           (rotary_emb): LlamaRotaryEmbedding()
#         )
#         (mlp): LlamaMLP(
#           (gate_proj): Linear(in_features=2560, out_features=10240, bias=False)
#           (up_proj): Linear(in_features=2560, out_features=10240, bias=False)
#           (down_proj): Linear(in_features=10240, out_features=2560, bias=False)
#           (act_fn): SiLU()
#         )
#         (input_layernorm): LlamaRMSNorm()
#         (post_attention_layernorm): LlamaRMSNorm()
#       )
#     )
#     (norm): LlamaRMSNorm()
#   )
#   (lm_head): Linear(in_features=2560, out_features=49152, bias=False)
# )


# Third Party
from peft import LoraConfig, TaskType, get_peft_model

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj", "k_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.05,
)

peft_model = get_peft_model(model, lora_config)

print(peft_model)


# PeftModelForCausalLM(
#   (base_model): LoraModel(
#     (model): LlamaForCausalLM(
#       (model): LlamaModel(
#         (embed_tokens): Embedding(49152, 2560, padding_idx=0)
#         (layers): ModuleList(
#           (0-31): 32 x LlamaDecoderLayer(
#             (self_attn): LlamaSdpaAttention(
#               (q_proj): lora.Linear(
#                 (base_layer): Linear(in_features=2560, out_features=2560, bias=True)
#                 (lora_dropout): ModuleDict(
#                   (default): Dropout(p=0.05, inplace=False)
#                 )
#                 (lora_A): ModuleDict(
#                   (default): Linear(in_features=2560, out_features=16, bias=False)
#                 )
#                 (lora_B): ModuleDict(
#                   (default): Linear(in_features=16, out_features=2560, bias=False)
#                 )
#                 (lora_embedding_A): ParameterDict()
#                 (lora_embedding_B): ParameterDict()
#               )
#               (k_proj): Linear(in_features=2560, out_features=2560, bias=True)
#               (v_proj): lora.Linear(
#                 (base_layer): Linear(in_features=2560, out_features=2560, bias=True)
#                 (lora_dropout): ModuleDict(
#                   (default): Dropout(p=0.05, inplace=False)
#                 )
#                 (lora_A): ModuleDict(
#                   (default): Linear(in_features=2560, out_features=16, bias=False)
#                 )
#                 (lora_B): ModuleDict(
#                   (default): Linear(in_features=16, out_features=2560, bias=False)
#                 )
#                 (lora_embedding_A): ParameterDict()
#                 (lora_embedding_B): ParameterDict()
#               )
#               (o_proj): Linear(in_features=2560, out_features=2560, bias=True)
#               (rotary_emb): LlamaRotaryEmbedding()
#             )
#             (mlp): LlamaMLP(
#               (gate_proj): Linear(in_features=2560, out_features=10240, bias=False)
#               (up_proj): Linear(in_features=2560, out_features=10240, bias=False)
#               (down_proj): Linear(in_features=10240, out_features=2560, bias=False)
#               (act_fn): SiLU()
#             )
#             (input_layernorm): LlamaRMSNorm()
#             (post_attention_layernorm): LlamaRMSNorm()
#           )
#         )
#         (norm): LlamaRMSNorm()
#       )
#       (lm_head): Linear(in_features=2560, out_features=49152, bias=False)
#     )
#   )
# )
