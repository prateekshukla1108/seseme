import os
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import LlamaTokenizer
import json
import logging
from pydantic import BaseModel
from torch import nn
from safetensors.torch import load_file as safetensors_load
import argparse
import gc

# Configure logging for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Optimize PyTorch for CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Model configuration using Pydantic for validation
class ModelArgs(BaseModel):
    backbone_flavor: str = "llama-1B"
    decoder_flavor: str = "llama-100M"
    text_vocab_size: int = 128256
    audio_vocab_size: int = 2051
    audio_num_codebooks: int = 32
    backbone_max_seq_len: int = 512
    embedding_dim: int = 2048
    decoder_embedding_dim: int = 1024
    backbone_num_layers: int = 16
    decoder_num_layers: int = 4
    backbone_num_heads: int = 16
    backbone_num_kv_heads: int = 4  # For grouped-query attention
    decoder_num_heads: int = 16
    decoder_num_kv_heads: int = 4
    backbone_intermediate_dim: int = 8192
    decoder_intermediate_dim: int = 8192
    backbone_head_dim: int = 128
    decoder_head_dim: int = 64

# Transformer layer with grouped-query attention
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int, head_dim: int, intermediate_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # Attention projections
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=True)
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=True)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, embed_dim)
        )
        
        # Normalization layers
        self.sa_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with grouped-query attention
        norm_x = self.sa_norm(x)
        q = self.q_proj(norm_x).view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(x.size(0), x.size(1), self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(x.size(0), x.size(1), self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Repeat k and v for grouped-query attention
        repeat_factor = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
        
        # Efficient scaled dot-product attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view_as(x)
        attn_output = self.out_proj(attn_output)
        
        # Residual connections
        x = x + attn_output
        x = x + self.mlp(self.mlp_norm(x))
        return x

# Main model class
class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.text_embeddings = nn.Embedding(config.text_vocab_size, config.embedding_dim)
        self.audio_embeddings = nn.Embedding(config.audio_vocab_size * config.audio_num_codebooks, config.embedding_dim)
        
        # Backbone transformer layers
        self.backbone = nn.ModuleList([
            TransformerLayer(
                embed_dim=config.embedding_dim,
                num_heads=config.backbone_num_heads,
                num_kv_heads=config.backbone_num_kv_heads,
                head_dim=config.backbone_head_dim,
                intermediate_dim=config.backbone_intermediate_dim
            ) for _ in range(config.backbone_num_layers)
        ])
        self.backbone_norm = nn.LayerNorm(config.embedding_dim)
        
        # Projection and decoder
        self.projection = nn.Linear(config.embedding_dim, config.decoder_embedding_dim, bias=True)
        self.decoder = nn.ModuleList([
            TransformerLayer(
                embed_dim=config.decoder_embedding_dim,
                num_heads=config.decoder_num_heads,
                num_kv_heads=config.decoder_num_kv_heads,
                head_dim=config.decoder_head_dim,
                intermediate_dim=config.decoder_intermediate_dim
            ) for _ in range(config.decoder_num_layers)
        ])
        self.decoder_norm = nn.LayerNorm(config.decoder_embedding_dim)
        
        # Output heads
        self.codebook0_head = nn.Linear(config.embedding_dim, config.audio_vocab_size, bias=True)
        self.audio_head = nn.Parameter(torch.randn(config.audio_num_codebooks - 1, config.decoder_embedding_dim, config.audio_vocab_size))

    def _embed_audio(self, index: int, sample: torch.Tensor) -> torch.Tensor:
        offset = index * self.config.audio_vocab_size
        return self.audio_embeddings(sample + offset)

# Base generator class
class Generator:
    def __init__(self, model: Model):
        self.model = model
        self.sample_rate = 24000
        self.frame_duration_ms = 10

    def generate(self, text: str, speaker: int, max_audio_length_ms: int) -> torch.Tensor:
        return torch.randn(int(self.sample_rate * max_audio_length_ms / 1000))

# CUDA-accelerated generator with memory optimizations
class CUDAGenerator(Generator):
    def __init__(self, model: Model):
        super().__init__(model)
        self.device = torch.device("cuda")
        self.model.eval()
        self.model.to(self.device, dtype=torch.float16)  # Mixed precision for memory efficiency
        self.tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

    def decode_audio_tokens(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        num_samples = int(self.sample_rate * (audio_tokens.size(1) * self.frame_duration_ms / 1000))
        return torch.randn(num_samples)  # Placeholder; replace with actual decoding logic if available

    def generate_frame(self, tokens: torch.Tensor, tokens_mask: torch.Tensor, input_pos: torch.Tensor,
                       temperature: float, topk: int, len_text: int) -> torch.Tensor:
        b, current_seq_len, _ = tokens.shape
        embeddings = torch.zeros(b, current_seq_len, self.model.config.embedding_dim,
                                 device=self.device, dtype=torch.float16)
        num_codebooks = self.model.config.audio_num_codebooks

        # Embed text tokens
        if len_text > 0 and current_seq_len == len_text:
            text_tokens = tokens[:, :len_text, num_codebooks]
            embeddings[:, :len_text] = self.model.text_embeddings(text_tokens)

        # Embed audio tokens
        if current_seq_len > len_text:
            i = current_seq_len - len_text - 1
            audio_token = tokens[:, len_text + i:current_seq_len, :].squeeze(1)
            audio_emb = sum(self.model._embed_audio(idx, audio_token[:, idx])
                            for idx in range(num_codebooks))
            embeddings[:, -1:] = audio_emb

        # Forward pass through backbone
        with torch.no_grad():
            x = embeddings
            for layer in self.model.backbone:
                x = layer(x)
                torch.cuda.empty_cache()  # Free memory after each layer
            x = self.model.backbone_norm(x)
            output = x[:, -1]

            # Generate first codebook token
            logits0 = self.codebook0_head(output)
            topk_val, topk_indices = torch.topk(logits0, k=topk, dim=-1)
            topk_probs = torch.softmax(topk_val, dim=-1)
            sampled_token0 = torch.multinomial(topk_probs, 1).squeeze(-1)

            # Prepare audio frame tokens
            audio_frame_tokens = torch.zeros(b, num_codebooks, dtype=torch.int32, device=self.device)
            audio_frame_tokens[:, 0] = sampled_token0

            # Generate remaining codebook tokens
            projected = self.model.projection(output)
            logits = torch.einsum('bd,ndv->bnv', projected, self.model.audio_head)
            sampled_tokens = torch.multinomial(
                torch.softmax(logits / temperature, dim=-1).view(-1, self.model.config.audio_vocab_size), 1
            ).view(b, -1)
            audio_frame_tokens[:, 1:] = sampled_tokens

        gc.collect()
        return audio_frame_tokens

    def generate(self, text: str, speaker: int, max_audio_length_ms: int) -> torch.Tensor:
        with torch.no_grad():
            torch.cuda.empty_cache()
            gc.collect()

            # Input validation
            if not text or not text.strip():
                raise ValueError("Text must be non-empty")
            if len(text) > 1000:
                raise ValueError("Text exceeds maximum length of 1000 characters")
            if not isinstance(speaker, int):
                raise ValueError("Speaker must be an integer")
            if max_audio_length_ms <= 0 or max_audio_length_ms > 60000:
                raise ValueError("max_audio_length_ms must be between 0 and 60000")

            logger.info(f"Generating audio for text: '{text}'")
            text_tokens = self.tokenizer.encode(text, return_tensors="pt")[0].to(self.device)
            b = 1
            num_codebooks = self.model.config.audio_num_codebooks
            num_audio_frames = min(
                (max_audio_length_ms // self.frame_duration_ms),
                self.model.config.backbone_max_seq_len - len(text_tokens)
            )

            # Initialize token tensor
            tokens = torch.zeros((b, self.model.config.backbone_max_seq_len, num_codebooks + 1),
                                 dtype=torch.int32, device=self.device)
            tokens[0, :len(text_tokens), num_codebooks] = text_tokens
            tokens_mask = torch.ones_like(tokens, dtype=torch.bool)
            current_seq_len = len(text_tokens)
            audio_tokens_list = []

            # Generate audio frames
            for _ in range(num_audio_frames):
                input_pos = torch.arange(current_seq_len, dtype=torch.int32, device=self.device).unsqueeze(0)
                frame_tokens = self.generate_frame(
                    tokens[:, :current_seq_len],
                    tokens_mask[:, :current_seq_len],
                    input_pos,
                    temperature=1.0,
                    topk=10,
                    len_text=len(text_tokens)
                )
                tokens[0, current_seq_len, :num_codebooks] = frame_tokens[0]
                audio_tokens_list.append(frame_tokens)
                current_seq_len += 1
                torch.cuda.empty_cache()

            # Stack audio tokens and decode
            audio_tokens = torch.stack(audio_tokens_list, dim=1).squeeze(0)
            waveform = self.decode_audio_tokens(audio_tokens)
            logger.info("Audio generation completed")
            return waveform

# Function to load the model
def load_csm_1b(device: str = "cuda") -> CUDAGenerator:
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    repo_id = "sesame/csm-1b"
    try:
        weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    except Exception as e:
        logger.error(f"Failed to download model files: {e}")
        raise

    state_dict = safetensors_load(weights_path)
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = ModelArgs(**config_dict)
    model = Model(config)
    model.load_state_dict(state_dict)
    logger.info("Model loaded successfully")
    torch.cuda.empty_cache()
    gc.collect()
    return CUDAGenerator(model)

# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio from text using the CSM-1B model")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID")
    parser.add_argument("--max_audio_length_ms", type=int, default=3000, help="Max audio length in milliseconds")
    parser.add_argument("--output_file", type=str, default="output.wav", help="Output audio file path")
    args = parser.parse_args()

    try:
        generator = load_csm_1b()
        with torch.no_grad():
            waveform = generator.generate(args.text, args.speaker, args.max_audio_length_ms)
        torchaudio.save(args.output_file, waveform.unsqueeze(0).float().cpu(), generator.sample_rate)
        logger.info(f"Audio saved to {args.output_file}")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        exit(1)
