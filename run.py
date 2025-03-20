import ctypes
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import LlamaTokenizer
import json
import os
import logging
from pydantic import BaseModel, ValidationError
from torch import nn
from safetensors.torch import load_file as safetensors_load
import argparse
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Model configuration class using Pydantic for validation
class ModelArgs(BaseModel):
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int
    backbone_max_seq_len: int = 1024  # Default value if not specified
    embedding_dim: int = 2048
    num_backbone_layers: int = 16
    num_decoder_layers: int = 4
    num_heads: int = 32
    intermediate_dim: int = 8192

# Transformer layer definition
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, intermediate_dim: int):
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        norm_x = self.sa_norm(x)
        q = self.q_proj(norm_x).view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
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

    def _embed_audio(self, index: int, sample: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(sample + index * self.config.audio_vocab_size)

# Base generator class
class Generator:
    def __init__(self, model: Model):
        self.model = model
        self.sample_rate = 24000
        self.frame_duration_ms = 10

    def generate(self, text: str, speaker: int, context: List, max_audio_length_ms: int) -> torch.Tensor:
        return torch.randn(int(self.sample_rate * max_audio_length_ms / 1000))

# CUDA-accelerated generator class using float32 exclusively
class CUDAGenerator(Generator):
    def __init__(self, model: Model):
        super().__init__(model)
        self.device = torch.device("cuda")
        self.model = model.to(self.device, dtype=torch.float32)  # Use float32 exclusively
        self.text_emb_weights = self.model.text_embeddings.weight.data.contiguous().to(self.device, dtype=torch.float32)
        self.audio_emb_weights = self.model.audio_embeddings.weight.data.contiguous().to(self.device, dtype=torch.float32)
        self.proj_weights = self.model.projection.weight.data.contiguous().to(self.device, dtype=torch.float32)
        self.proj_bias = self.model.projection.bias.data.contiguous().to(self.device, dtype=torch.float32)
        self.c0_head_weights = self.model.codebook0_head.weight.data.contiguous().to(self.device, dtype=torch.float32)
        self.c0_head_bias = self.model.codebook0_head.bias.data.contiguous().to(self.device, dtype=torch.float32)
        self.audio_head_weights = self.model.audio_head.data.contiguous().to(self.device, dtype=torch.float32)
        self.audio_head_biases = torch.zeros(
            self.model.config.audio_num_codebooks - 1,
            self.model.config.audio_vocab_size,
            device=self.device,
            dtype=torch.float32
        )
        self.backbone_q_proj_weights = [
            layer.q_proj.weight.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.backbone
        ]
        self.backbone_k_proj_weights = [
            layer.k_proj.weight.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.backbone
        ]
        self.backbone_v_proj_weights = [
            layer.v_proj.weight.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.backbone
        ]
        self.backbone_out_proj_weights = [
            layer.out_proj.weight.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.backbone
        ]
        self.backbone_q_proj_biases = [
            layer.q_proj.bias.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.backbone
        ]
        self.backbone_k_proj_biases = [
            layer.k_proj.bias.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.backbone
        ]
        self.backbone_v_proj_biases = [
            layer.v_proj.bias.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.backbone
        ]
        self.backbone_out_proj_biases = [
            layer.out_proj.bias.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.backbone
        ]
        self.backbone_mlp_w1_weights = [
            layer.mlp[0].weight.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.backbone
        ]
        self.backbone_mlp_w2_weights = [
            layer.mlp[2].weight.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.backbone
        ]
        self.backbone_mlp_w1_biases = [
            layer.mlp[0].bias.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.backbone
        ]
        self.backbone_mlp_w2_biases = [
            layer.mlp[2].bias.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.backbone
        ]
        self.backbone_sa_norm_weights = [
            layer.sa_norm.weight.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.backbone
        ]
        self.backbone_mlp_norm_weights = [
            layer.mlp_norm.weight.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.backbone
        ]
        self.backbone_sa_norm_biases = [
            layer.sa_norm.bias.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.backbone
        ]
        self.backbone_mlp_norm_biases = [
            layer.mlp_norm.bias.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.backbone
        ]
        self.decoder_q_proj_weights = [
            layer.q_proj.weight.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.decoder
        ]
        self.decoder_k_proj_weights = [
            layer.k_proj.weight.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.decoder
        ]
        self.decoder_v_proj_weights = [
            layer.v_proj.weight.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.decoder
        ]
        self.decoder_out_proj_weights = [
            layer.out_proj.weight.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.decoder
        ]
        self.decoder_q_proj_biases = [
            layer.q_proj.bias.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.decoder
        ]
        self.decoder_k_proj_biases = [
            layer.k_proj.bias.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.decoder
        ]
        self.decoder_v_proj_biases = [
            layer.v_proj.bias.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.decoder
        ]
        self.decoder_out_proj_biases = [
            layer.out_proj.bias.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.decoder
        ]
        self.decoder_mlp_w1_weights = [
            layer.mlp[0].weight.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.decoder
        ]
        self.decoder_mlp_w2_weights = [
            layer.mlp[2].weight.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.decoder
        ]
        self.decoder_mlp_w1_biases = [
            layer.mlp[0].bias.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.decoder
        ]
        self.decoder_mlp_w2_biases = [
            layer.mlp[2].bias.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.decoder
        ]
        self.decoder_sa_norm_weights = [
            layer.sa_norm.weight.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.decoder
        ]
        self.decoder_mlp_norm_weights = [
            layer.mlp_norm.weight.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.decoder
        ]
        self.decoder_sa_norm_biases = [
            layer.sa_norm.bias.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.decoder
        ]
        self.decoder_mlp_norm_biases = [
            layer.mlp_norm.bias.data.contiguous().to(self.device, dtype=torch.float32)
            for layer in self.model.decoder
        ]
        self.decoder_norm_weights = self.model.decoder_norm.weight.data.contiguous().to(self.device, dtype=torch.float32)
        self.decoder_norm_biases = self.model.decoder_norm.bias.data.contiguous().to(self.device, dtype=torch.float32)
        self.backbone_norm_weights = self.model.backbone_norm.weight.data.contiguous().to(self.device, dtype=torch.float32)
        self.backbone_norm_biases = self.model.backbone_norm.bias.data.contiguous().to(self.device, dtype=torch.float32)

    def decode_audio_tokens(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Decoding of audio tokens into waveform is not implemented. Integrate a vocoder (e.g., HiFi-GAN) here.")

    def generate_frame(self, tokens: torch.Tensor, tokens_mask: torch.Tensor, input_pos: torch.Tensor,
                       temperature: float, topk: int, len_text: int) -> torch.Tensor:
        b, current_seq_len, _ = tokens.shape
        embeddings = torch.zeros(b, current_seq_len, self.model.config.embedding_dim, device=self.device, dtype=torch.float32)
        if len_text > 0 and current_seq_len == len_text:
            text_tokens = tokens[:, :len_text, self.model.config.audio_num_codebooks]
            embeddings[:, :len_text, :] = nn.functional.embedding(text_tokens, self.text_emb_weights)
        if current_seq_len > len_text:
            i = current_seq_len - len_text - 1
            audio_token = tokens[:, len_text + i:current_seq_len, :].squeeze(1)
            audio_emb = sum(self.model._embed_audio(idx, audio_token[:, idx]) for idx in range(self.model.config.audio_num_codebooks))
            embeddings[:, -1:, :] = audio_emb
        mask = torch.empty(current_seq_len, current_seq_len, dtype=torch.bool, device=self.device)
        # Call the CUDA kernel (casting mask pointer remains unchanged)
        lib.create_causal_mask_cuda(ctypes.c_int(current_seq_len), ctypes.cast(mask.data_ptr(), ctypes.POINTER(ctypes.c_bool)))
        x = embeddings
        for layer_idx in range(self.model.config.num_backbone_layers):
            output = torch.empty_like(x)
            lib.transformer_decoder_forward_cuda(
                ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(b),
                ctypes.c_int(current_seq_len),
                ctypes.c_int(self.model.config.embedding_dim),
                ctypes.c_int(self.model.config.num_heads),
                ctypes.c_int(self.model.config.intermediate_dim),
                ctypes.cast(self.backbone_q_proj_weights[layer_idx].data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(self.backbone_q_proj_biases[layer_idx].data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(self.backbone_k_proj_weights[layer_idx].data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(self.backbone_k_proj_biases[layer_idx].data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(self.backbone_v_proj_weights[layer_idx].data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(self.backbone_v_proj_biases[layer_idx].data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(self.backbone_out_proj_weights[layer_idx].data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(self.backbone_out_proj_biases[layer_idx].data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(self.backbone_mlp_w1_weights[layer_idx].data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(self.backbone_mlp_w1_biases[layer_idx].data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(self.backbone_mlp_w2_weights[layer_idx].data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(self.backbone_mlp_w2_biases[layer_idx].data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(self.backbone_sa_norm_weights[layer_idx].data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(self.backbone_sa_norm_biases[layer_idx].data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(self.backbone_mlp_norm_weights[layer_idx].data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(self.backbone_mlp_norm_biases[layer_idx].data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(mask.data_ptr(), ctypes.POINTER(ctypes.c_bool)),
                ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_float))
            )
            x = output
        x = nn.functional.layer_norm(x, (self.model.config.embedding_dim,), weight=self.backbone_norm_weights, bias=self.backbone_norm_biases)
        output = x[:, -1, :]
        logits0 = torch.empty(b, self.model.config.audio_vocab_size, device=self.device, dtype=torch.float32)
        lib.linear_forward_cuda(
            ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(b),
            ctypes.c_int(1),
            ctypes.c_int(self.model.config.embedding_dim),
            ctypes.c_int(self.model.config.audio_vocab_size),
            ctypes.cast(self.c0_head_weights.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(self.c0_head_bias.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(logits0.data_ptr(), ctypes.POINTER(ctypes.c_float))
        )
        projected = torch.empty(b, 1024, device=self.device, dtype=torch.float32)
        lib.linear_forward_cuda(
            ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(b),
            ctypes.c_int(1),
            ctypes.c_int(self.model.config.embedding_dim),
            ctypes.c_int(1024),
            ctypes.cast(self.proj_weights.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(self.proj_bias.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(projected.data_ptr(), ctypes.POINTER(ctypes.c_float))
        )
        audio_frame_tokens = torch.zeros(b, self.model.config.audio_num_codebooks, dtype=torch.int32, device=self.device)
        sampled_token0 = torch.empty(b, dtype=torch.int32, device=self.device)
        lib.sample_topk_cuda(
            ctypes.cast(logits0.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(b),
            ctypes.c_float(temperature),
            ctypes.c_int(topk),
            ctypes.c_int(self.model.config.audio_vocab_size),
            ctypes.cast(sampled_token0.data_ptr(), ctypes.POINTER(ctypes.c_int))
        )
        audio_frame_tokens[:, 0] = sampled_token0
        logits = torch.einsum('bd,ndv->bnv', projected, self.audio_head_weights)
        sampled_tokens = torch.multinomial(
            torch.softmax(logits / temperature, dim=-1).view(-1, self.model.config.audio_vocab_size), 1
        ).view(b, -1)
        audio_frame_tokens[:, 1:] = sampled_tokens
        return audio_frame_tokens

    def generate(self, text: str, speaker: int, context: List, max_audio_length_ms: int) -> torch.Tensor:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")
        if len(text) > 1000:
            raise ValueError("Text exceeds maximum length of 1000 characters")
        if not isinstance(speaker, int):
            raise ValueError("Speaker must be an integer")
        if not isinstance(context, list):
            raise ValueError("Context must be a list")
        if not isinstance(max_audio_length_ms, int) or max_audio_length_ms <= 0:
            raise ValueError("max_audio_length_ms must be a positive integer")
        if max_audio_length_ms > 60000:
            raise ValueError("max_audio_length_ms exceeds limit of 60000 ms")
        logger.info(f"Starting audio generation for text: '{text}'")
        text_tokens = tokenize_text(text, self.model.config.text_vocab_size)
        if not text_tokens:
            logger.warning("No tokens generated from text")
            return torch.zeros(0, device=self.device)
        b = 1
        num_codebooks = self.model.config.audio_num_codebooks
        frame_duration_ms = self.frame_duration_ms
        num_audio_frames = min(
            (max_audio_length_ms // frame_duration_ms),
            self.model.config.backbone_max_seq_len - len(text_tokens)
        )
        tokens = torch.zeros((b, self.model.config.backbone_max_seq_len, num_codebooks + 1), dtype=torch.int32, device=self.device)
        tokens[0, :len(text_tokens), num_codebooks] = torch.tensor(
            text_tokens[:self.model.config.backbone_max_seq_len], dtype=torch.int32, device=self.device
        )
        tokens_mask = torch.ones_like(tokens, dtype=torch.bool)
        current_seq_len = len(text_tokens)
        audio_tokens_list = []
        try:
            for _ in range(num_audio_frames):
                input_pos = torch.arange(current_seq_len, dtype=torch.int32, device=self.device).unsqueeze(0)
                audio_frame_tokens = self.generate_frame(
                    tokens[:, :current_seq_len, :], tokens_mask[:, :current_seq_len, :], input_pos, 1.0, 10, len(text_tokens)
                )
                tokens[0, current_seq_len, :num_codebooks] = audio_frame_tokens[0]
                audio_tokens_list.append(audio_frame_tokens)
                current_seq_len += 1
        except Exception as e:
            logger.error(f"Error during frame generation: {str(e)}")
            raise
        audio_tokens = torch.stack(audio_tokens_list, dim=1).squeeze(0)
        waveform = self.decode_audio_tokens(audio_tokens)
        logger.info("Audio generation completed")
        return waveform

# Tokenization function with proper error handling
def tokenize_text(text: str, text_vocab_size: int) -> List[int]:
    try:
        # Use LlamaTokenizer since the model is based on LLaMA
        tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", trust_remote_code=True)
        tokens = tokenizer.encode(text, return_tensors="pt")[0].tolist()
        unk_token_id = tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else 0
        # Ensure tokens are within valid range [0, text_vocab_size - 1]
        tokens = [t if 0 <= t < text_vocab_size else unk_token_id for t in tokens]
        return tokens
    except Exception as e:
        logger.error(f"Failed to load or use tokenizer: {e}")
        raise

# Model loading function
def load_csm_1b(device: str = "cuda") -> CUDAGenerator:
    if not torch.cuda.is_available():
        logger.error("CUDA is not available")
        raise RuntimeError("CUDA not available")
    repo_id = "sesame/csm-1b"  # Adjust this to your actual model repository
    try:
        weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    except Exception as e:
        logger.error(f"Failed to download model files: {e}")
        raise
    state_dict = safetensors_load(weights_path)
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    if 'CSM_BACKBONE_FLAVOR' in os.environ:
        config_dict['backbone_flavor'] = os.environ['CSM_BACKBONE_FLAVOR']
    try:
        config = ModelArgs(**config_dict)
    except ValidationError as e:
        logger.error(f"Invalid configuration: {e}")
        raise
    model = Model(config)
    model.load_state_dict(state_dict, strict=False)
    logger.info("Model loaded successfully")
    return CUDAGenerator(model)

# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio from text using CSM-1B model")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID")
    parser.add_argument("--context", type=str, default="", help="Context (not implemented)")
    parser.add_argument("--max_audio_length_ms", type=int, default=5000, help="Maximum audio length in milliseconds")
    parser.add_argument("--output_file", type=str, default="output.wav", help="Output audio file path")
    args = parser.parse_args()
    if not os.path.exists('./libtts_kernels.so'):
        logger.error("CUDA kernel library 'libtts_kernels.so' not found. Ensure 'libtts_kernels.so' is in the same directory.")
        exit(1)
    lib = ctypes.CDLL('./libtts_kernels.so')
    # Set argument types for functions expecting float (32-bit) pointers using c_float:
    lib.create_causal_mask_cuda.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_bool)]
    lib.index_causal_mask_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_bool)
    ]
    lib.sample_topk_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
    ]
    lib.embedding_forward_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
    ]
    lib.linear_forward_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
    ]
    lib.transformer_decoder_forward_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_float)
    ]
    try:
        generator = load_csm_1b()
        waveform = generator.generate(args.text, args.speaker, [], args.max_audio_length_ms)
        torchaudio.save(args.output_file, waveform.unsqueeze(0).float().cpu(), generator.sample_rate)
        logger.info(f"Audio saved to {args.output_file}")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        exit(1)

