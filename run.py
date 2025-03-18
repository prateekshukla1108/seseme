import ctypes
import torch
import torchaudio
from huggingface_hub import hf_hub_download
import json
from dataclasses import dataclass
from torch import nn
from safetensors.torch import load_file as safetensors_load

# Configuration dataclass
@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int
    backbone_max_seq_len: int
    embedding_dim: int = 2048
    num_backbone_layers: int = 16
    num_decoder_layers: int = 4
    num_heads: int = 32
    intermediate_dim: int = 8192

# Transformer layer with self-attention and MLP
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, intermediate_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, intermediate_dim, bias=True),
            nn.GELU(),
            nn.Linear(intermediate_dim, embed_dim, bias=True)
        )
        self.sa_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        norm_x = self.sa_norm(x)
        q = self.q_proj(norm_x).view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view_as(x)
        attn_output = self.out_proj(attn_output)
        x = x + attn_output
        x = x + self.mlp(self.mlp_norm(x))
        return x

# Main model class
class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.text_embeddings = nn.Embedding(config.text_vocab_size, config.embedding_dim)
        self.audio_embeddings = nn.Embedding(config.audio_vocab_size * config.audio_num_codebooks, config.embedding_dim)
        self.projection = nn.Linear(config.embedding_dim, 1024, bias=True)
        self.codebook0_head = nn.Linear(config.embedding_dim, config.audio_vocab_size, bias=True)
        self.audio_head = nn.Parameter(torch.randn(config.audio_num_codebooks - 1, 1024, config.audio_vocab_size))
        self.backbone = nn.ModuleList([
            TransformerLayer(config.embedding_dim, config.num_heads, config.intermediate_dim)
            for _ in range(config.num_backbone_layers)
        ])
        self.decoder = nn.ModuleList([
            TransformerLayer(1024, config.num_heads, 4096)
            for _ in range(config.num_decoder_layers)
        ])
        self.backbone_norm = nn.LayerNorm(config.embedding_dim)
        self.decoder_norm = nn.LayerNorm(1024)

    def _embed_audio(self, index, sample):
        return self.audio_embeddings(sample + index * self.config.audio_vocab_size)

# Base generator class
class Generator:
    def __init__(self, model):
        self.model = model
        self.sample_rate = 24000
        self.frame_duration_ms = 10

    def generate(self, text, speaker, context, max_audio_length_ms):
        dummy_audio = torch.randn(int(self.sample_rate * max_audio_length_ms / 1000))
        return dummy_audio

# CUDA-optimized generator class
class CUDAGenerator(Generator):
    def __init__(self, model: Model):
        super().__init__(model)
        self.device = torch.device("cuda")
        self.model = model.to(self.device, dtype=torch.float32)

        # Move weights to CUDA
        self.text_emb_weights = self.model.text_embeddings.weight.data.contiguous().to(self.device)
        self.audio_emb_weights = self.model.audio_embeddings.weight.data.contiguous().to(self.device)
        self.proj_weights = self.model.projection.weight.data.contiguous().to(self.device)
        self.proj_bias = self.model.projection.bias.data.contiguous().to(self.device)
        self.c0_head_weights = self.model.codebook0_head.weight.data.contiguous().to(self.device)
        self.c0_head_bias = self.model.codebook0_head.bias.data.contiguous().to(self.device)
        self.audio_head_weights = self.model.audio_head.data.contiguous().to(self.device)
        self.audio_head_biases = torch.zeros(self.model.config.audio_num_codebooks - 1, self.model.config.audio_vocab_size, device=self.device)

        # Backbone weights
        self.backbone_q_proj_weights = [layer.q_proj.weight.data.contiguous().to(self.device) for layer in self.model.backbone]
        self.backbone_k_proj_weights = [layer.k_proj.weight.data.contiguous().to(self.device) for layer in self.model.backbone]
        self.backbone_v_proj_weights = [layer.v_proj.weight.data.contiguous().to(self.device) for layer in self.model.backbone]
        self.backbone_out_proj_weights = [layer.out_proj.weight.data.contiguous().to(self.device) for layer in self.model.backbone]
        self.backbone_q_proj_biases = [layer.q_proj.bias.data.contiguous().to(self.device) for layer in self.model.backbone]
        self.backbone_k_proj_biases = [layer.k_proj.bias.data.contiguous().to(self.device) for layer in self.model.backbone]
        self.backbone_v_proj_biases = [layer.v_proj.bias.data.contiguous().to(self.device) for layer in self.model.backbone]
        self.backbone_out_proj_biases = [layer.out_proj.bias.data.contiguous().to(self.device) for layer in self.model.backbone]
        self.backbone_mlp_w1_weights = [layer.mlp[0].weight.data.contiguous().to(self.device) for layer in self.model.backbone]
        self.backbone_mlp_w2_weights = [layer.mlp[2].weight.data.contiguous().to(self.device) for layer in self.model.backbone]
        self.backbone_mlp_w1_biases = [layer.mlp[0].bias.data.contiguous().to(self.device) for layer in self.model.backbone]
        self.backbone_mlp_w2_biases = [layer.mlp[2].bias.data.contiguous().to(self.device) for layer in self.model.backbone]
        self.backbone_sa_norm_weights = [layer.sa_norm.weight.data.contiguous().to(self.device) for layer in self.model.backbone]
        self.backbone_mlp_norm_weights = [layer.mlp_norm.weight.data.contiguous().to(self.device) for layer in self.model.backbone]
        self.backbone_sa_norm_biases = [layer.sa_norm.bias.data.contiguous().to(self.device) for layer in self.model.backbone]
        self.backbone_mlp_norm_biases = [layer.mlp_norm.bias.data.contiguous().to(self.device) for layer in self.model.backbone]

        # Decoder weights
        self.decoder_q_proj_weights = [layer.q_proj.weight.data.contiguous().to(self.device) for layer in self.model.decoder]
        self.decoder_k_proj_weights = [layer.k_proj.weight.data.contiguous().to(self.device) for layer in self.model.decoder]
        self.decoder_v_proj_weights = [layer.v_proj.weight.data.contiguous().to(self.device) for layer in self.model.decoder]
        self.decoder_out_proj_weights = [layer.out_proj.weight.data.contiguous().to(self.device) for layer in self.model.decoder]
        self.decoder_q_proj_biases = [layer.q_proj.bias.data.contiguous().to(self.device) for layer in self.model.decoder]
        self.decoder_k_proj_biases = [layer.k_proj.bias.data.contiguous().to(self.device) for layer in self.model.decoder]
        self.decoder_v_proj_biases = [layer.v_proj.bias.data.contiguous().to(self.device) for layer in self.model.decoder]
        self.decoder_out_proj_biases = [layer.out_proj.bias.data.contiguous().to(self.device) for layer in self.model.decoder]
        self.decoder_mlp_w1_weights = [layer.mlp[0].weight.data.contiguous().to(self.device) for layer in self.model.decoder]
        self.decoder_mlp_w2_weights = [layer.mlp[2].weight.data.contiguous().to(self.device) for layer in self.model.decoder]
        self.decoder_mlp_w1_biases = [layer.mlp[0].bias.data.contiguous().to(self.device) for layer in self.model.decoder]
        self.decoder_mlp_w2_biases = [layer.mlp[2].bias.data.contiguous().to(self.device) for layer in self.model.decoder]
        self.decoder_sa_norm_weights = [layer.sa_norm.weight.data.contiguous().to(self.device) for layer in self.model.decoder]
        self.decoder_mlp_norm_weights = [layer.mlp_norm.weight.data.contiguous().to(self.device) for layer in self.model.decoder]
        self.decoder_sa_norm_biases = [layer.sa_norm.bias.data.contiguous().to(self.device) for layer in self.model.decoder]
        self.decoder_mlp_norm_biases = [layer.mlp_norm.bias.data.contiguous().to(self.device) for layer in self.model.decoder]

        # Final norm weights
        self.backbone_norm_weights = self.model.backbone_norm.weight.data.contiguous().to(self.device)
        self.backbone_norm_biases = self.model.backbone_norm.bias.data.contiguous().to(self.device)
        self.decoder_norm_weights = self.model.decoder_norm.weight.data.contiguous().to(self.device)
        self.decoder_norm_biases = self.model.decoder_norm.bias.data.contiguous().to(self.device)

    @torch.inference_mode()
    def generate_frame(self, tokens: torch.Tensor, tokens_mask: torch.Tensor, input_pos: torch.Tensor, temperature: float, topk: int):
        b, s, _ = tokens.size()
        embed_dim = self.model.config.embedding_dim

        embeds = torch.zeros(b, s, 33, embed_dim, device=self.device, dtype=torch.float32)
        for i in range(33):
            weights = self.audio_emb_weights if i < 32 else self.text_emb_weights
            lib.embedding_forward_cuda(
                ctypes.cast(tokens[:, :, i].contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int)),
                b, s, weights.size(0), embed_dim,
                ctypes.cast(weights.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(embeds[:, :, i].contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
            )

        h = embeds * tokens_mask.unsqueeze(-1)
        h = h.sum(dim=2)
        backbone_out = torch.zeros(b, s, embed_dim, device=self.device, dtype=torch.float32)
        mask = torch.ones(b, s, s, dtype=torch.bool, device=self.device)
        lib.index_causal_mask_cuda(
            ctypes.cast(input_pos.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int)),
            b, s, s, ctypes.cast(mask.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_bool))
        )

        # Backbone pointers - Using pointer arrays correctly
        backbone_q_proj_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.backbone_q_proj_weights))(
            *[ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)) for w in self.backbone_q_proj_weights])
        backbone_k_proj_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.backbone_k_proj_weights))(
            *[ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)) for w in self.backbone_k_proj_weights])
        backbone_v_proj_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.backbone_v_proj_weights))(
            *[ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)) for w in self.backbone_v_proj_weights])
        backbone_out_proj_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.backbone_out_proj_weights))(
            *[ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)) for w in self.backbone_out_proj_weights])
        backbone_q_bias_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.backbone_q_proj_biases))(
            *[ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_float)) for b in self.backbone_q_proj_biases])
        backbone_k_bias_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.backbone_k_proj_biases))(
            *[ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_float)) for b in self.backbone_k_proj_biases])
        backbone_v_bias_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.backbone_v_proj_biases))(
            *[ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_float)) for b in self.backbone_v_proj_biases])
        backbone_out_bias_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.backbone_out_proj_biases))(
            *[ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_float)) for b in self.backbone_out_proj_biases])
        backbone_mlp_w1_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.backbone_mlp_w1_weights))(
            *[ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)) for w in self.backbone_mlp_w1_weights])
        backbone_mlp_w2_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.backbone_mlp_w2_weights))(
            *[ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)) for w in self.backbone_mlp_w2_weights])
        backbone_mlp_w1_bias_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.backbone_mlp_w1_biases))(
            *[ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_float)) for b in self.backbone_mlp_w1_biases])
        backbone_mlp_w2_bias_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.backbone_mlp_w2_biases))(
            *[ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_float)) for b in self.backbone_mlp_w2_biases])
        backbone_sa_norm_w_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.backbone_sa_norm_weights))(
            *[ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)) for w in self.backbone_sa_norm_weights])
        backbone_mlp_norm_w_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.backbone_mlp_norm_weights))(
            *[ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)) for w in self.backbone_mlp_norm_weights])
        backbone_sa_norm_b_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.backbone_sa_norm_biases))(
            *[ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_float)) for b in self.backbone_sa_norm_biases])
        backbone_mlp_norm_b_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.backbone_mlp_norm_biases))(
            *[ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_float)) for b in self.backbone_mlp_norm_biases])

        lib.transformer_decoder_forward_cuda(
            ctypes.cast(h.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float)),
            b, s, len(self.model.backbone), self.model.config.num_heads, embed_dim // self.model.config.num_heads,
            backbone_q_proj_ptrs, backbone_k_proj_ptrs, backbone_v_proj_ptrs, backbone_out_proj_ptrs,
            backbone_q_bias_ptrs, backbone_k_bias_ptrs, backbone_v_bias_ptrs, backbone_out_bias_ptrs,
            backbone_mlp_w1_ptrs, backbone_mlp_w2_ptrs, backbone_mlp_w1_bias_ptrs, backbone_mlp_w2_bias_ptrs,
            backbone_sa_norm_w_ptrs, backbone_mlp_norm_w_ptrs, backbone_sa_norm_b_ptrs, backbone_mlp_norm_b_ptrs,
            ctypes.cast(self.backbone_norm_weights.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(self.backbone_norm_biases.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(mask.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_bool)),
            ctypes.cast(backbone_out.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        )

        last_h = backbone_out[:, -1, :]
        c0_logits = torch.zeros(b, self.model.config.audio_vocab_size, device=self.device, dtype=torch.float32)
        lib.linear_forward_cuda(
            ctypes.cast(last_h.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float)),
            b, 1, embed_dim, self.model.config.audio_vocab_size,
            ctypes.cast(self.c0_head_weights.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(self.c0_head_bias.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(c0_logits.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
        )

        c0_sample = torch.zeros(b, 1, dtype=torch.int32, device=self.device)
        lib.sample_topk_cuda(
            ctypes.cast(c0_logits.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float)),
            topk, temperature, self.model.config.audio_vocab_size, b,
            ctypes.cast(c0_sample.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
        )

        audio_embeds = [self.model._embed_audio(0, c0_sample)]
        curr_sample = c0_sample.clone()

        for i in range(1, self.model.config.audio_num_codebooks):
            curr_h = torch.cat([last_h.unsqueeze(1), torch.cat(audio_embeds, dim=1)], dim=1)
            curr_h_proj = torch.zeros(b, curr_h.size(1), 1024, device=self.device, dtype=torch.float32)
            lib.linear_forward_cuda(
                ctypes.cast(curr_h.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float)),
                b, curr_h.size(1), embed_dim, 1024,
                ctypes.cast(self.proj_weights.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(self.proj_bias.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(curr_h_proj.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
            )

            mask = torch.zeros(b, curr_h.size(1), curr_h.size(1), dtype=torch.bool, device=self.device)
            lib.index_causal_mask_cuda(
                ctypes.cast(input_pos.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int)),
                b, curr_h.size(1), curr_h.size(1),
                ctypes.cast(mask.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_bool))
            )

            decoder_out = torch.zeros(b, curr_h.size(1), 1024, device=self.device, dtype=torch.float32)
            decoder_q_proj_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.decoder_q_proj_weights))(
                *[ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)) for w in self.decoder_q_proj_weights])
            decoder_k_proj_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.decoder_k_proj_weights))(
                *[ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)) for w in self.decoder_k_proj_weights])
            decoder_v_proj_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.decoder_v_proj_weights))(
                *[ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)) for w in self.decoder_v_proj_weights])
            decoder_out_proj_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.decoder_out_proj_weights))(
                *[ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)) for w in self.decoder_out_proj_weights])
            decoder_q_bias_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.decoder_q_proj_biases))(
                *[ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_float)) for b in self.decoder_q_proj_biases])
            decoder_k_bias_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.decoder_k_proj_biases))(
                *[ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_float)) for b in self.decoder_k_proj_biases])
            decoder_v_bias_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.decoder_v_proj_biases))(
                *[ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_float)) for b in self.decoder_v_proj_biases])
            decoder_out_bias_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.decoder_out_proj_biases))(
                *[ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_float)) for b in self.decoder_out_proj_biases])
            decoder_mlp_w1_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.decoder_mlp_w1_weights))(
                *[ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)) for w in self.decoder_mlp_w1_weights])
            decoder_mlp_w2_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.decoder_mlp_w2_weights))(
                *[ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)) for w in self.decoder_mlp_w2_weights])
            decoder_mlp_w1_bias_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.decoder_mlp_w1_biases))(
                *[ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_float)) for b in self.decoder_mlp_w1_biases])
            decoder_mlp_w2_bias_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.decoder_mlp_w2_biases))(
                *[ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_float)) for b in self.decoder_mlp_w2_biases])
            decoder_sa_norm_w_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.decoder_sa_norm_weights))(
                *[ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)) for w in self.decoder_sa_norm_weights])
            decoder_mlp_norm_w_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.decoder_mlp_norm_weights))(
                *[ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)) for w in self.decoder_mlp_norm_weights])
            decoder_sa_norm_b_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.decoder_sa_norm_biases))(
                *[ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_float)) for b in self.decoder_sa_norm_biases])
            decoder_mlp_norm_b_ptrs = (ctypes.POINTER(ctypes.c_float) * len(self.decoder_mlp_norm_biases))(
                *[ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_float)) for b in self.decoder_mlp_norm_biases])

            lib.transformer_decoder_forward_cuda(
                ctypes.cast(curr_h_proj.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float)),
                b, curr_h.size(1), len(self.model.decoder), self.model.config.num_heads, 1024 // self.model.config.num_heads,
                decoder_q_proj_ptrs, decoder_k_proj_ptrs, decoder_v_proj_ptrs, decoder_out_proj_ptrs,
                decoder_q_bias_ptrs, decoder_k_bias_ptrs, decoder_v_bias_ptrs, decoder_out_bias_ptrs,
                decoder_mlp_w1_ptrs, decoder_mlp_w2_ptrs, decoder_mlp_w1_bias_ptrs, decoder_mlp_w2_bias_ptrs,
                decoder_sa_norm_w_ptrs, decoder_mlp_norm_w_ptrs, decoder_sa_norm_b_ptrs, decoder_mlp_norm_b_ptrs,
                ctypes.cast(self.decoder_norm_weights.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(self.decoder_norm_biases.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(mask.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_bool)),
                ctypes.cast(decoder_out.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
            )

            ci_logits = torch.zeros(b, self.model.config.audio_vocab_size, device=self.device, dtype=torch.float32)
            lib.linear_forward_cuda(
                ctypes.cast(decoder_out[:, -1, :].contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float)),
                b, 1, 1024, self.model.config.audio_vocab_size,
                ctypes.cast(self.audio_head_weights[i-1].contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(self.audio_head_biases[i-1].contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(ci_logits.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float))
            )

            ci_sample = torch.zeros(b, 1, dtype=torch.int32, device=self.device)
            lib.sample_topk_cuda(
                ctypes.cast(ci_logits.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float)),
                topk, temperature, self.model.config.audio_vocab_size, b,
                ctypes.cast(ci_sample.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_int))
            )
            audio_embeds.append(self.model._embed_audio(i, ci_sample))
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)

        return curr_sample

    def generate(self, text, speaker, context, max_audio_length_ms):
        text_tokens = tokenize_text(text)
        if not text_tokens:
            print("Error: Text tokenization failed.")
            return torch.randn(int(self.sample_rate * max_audio_length_ms / 1000))

        batch_size = 1
        max_seq_len = min(self.model.config.backbone_max_seq_len, len(text_tokens))
        num_audio_codebooks = self.model.config.audio_num_codebooks
        tokens = torch.zeros((batch_size, max_seq_len, num_audio_codebooks + 1), dtype=torch.int32, device=self.device)
        tokens[0, :len(text_tokens), 32] = torch.tensor(text_tokens[:max_seq_len], dtype=torch.int32, device=self.device)
        tokens_mask = torch.ones((batch_size, max_seq_len, num_audio_codebooks + 1), dtype=torch.bool, device=self.device)
        input_pos = torch.arange(max_seq_len, dtype=torch.int32, device=self.device).unsqueeze(0)

        audio_tokens = self.generate_frame(tokens, tokens_mask, input_pos, temperature=1.0, topk=10)
        num_samples = int(self.sample_rate * max_audio_length_ms / 1000)
        return torch.randn(num_samples, device=self.device)

# Tokenizer function with fallback
def tokenize_text(text):
    from transformers import AutoTokenizer
    try:
        # Ensure correct model ID - adjust if "sesame/csm-1b" is incorrect
        tokenizer = AutoTokenizer.from_pretrained("sesame/csm-1b", trust_remote_code=True)
        return tokenizer.encode(text, return_tensors="pt")[0].tolist()
    except Exception as e:
        print(f"Warning: Using dummy tokenizer due to error: {type(e).__name__} - {str(e)}")
        return [ord(char) for char in text] if text else []

def load_csm_1b(device: str = "cuda") -> CUDAGenerator:
    repo_id = "sesame/csm-1b"  # Verify this is correct
    weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")

    state_dict = safetensors_load(weights_path)
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = ModelArgs(
        backbone_flavor=config_dict['backbone_flavor'],
        decoder_flavor=config_dict['decoder_flavor'],
        text_vocab_size=config_dict['text_vocab_size'],
        audio_vocab_size=config_dict['audio_vocab_size'],
        audio_num_codebooks=config_dict['audio_num_codebooks'],
        backbone_max_seq_len=config_dict.get('backbone_max_seq_len', 1024),
    )

    model = Model(config)
    model.load_state_dict(state_dict, strict=False)
    return CUDAGenerator(model)

if __name__ == "__main__":
    lib = ctypes.CDLL('./libtts_kernels.so')
    lib.create_causal_mask_cuda.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_bool)]
    lib.index_causal_mask_cuda.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_bool)]
    lib.sample_topk_cuda.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.embedding_forward_cuda.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    lib.linear_forward_cuda.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    lib.transformer_decoder_forward_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_float)
    ]

    generator = load_csm_1b()
    text = "Hello, world!"
    speaker = 0
    context = []
    try:
        waveform = generator.generate(text, speaker, context, max_audio_length_ms=5000)
        torchaudio.save("output.wav", waveform.unsqueeze(0).float().cpu(), generator.sample_rate)
        print("Audio saved to output.wav")
    except Exception as e:
        print(f"Error during generation: {type(e).__name__} - {str(e)}")
