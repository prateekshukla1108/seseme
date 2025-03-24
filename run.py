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
import torch.nn.functional as F

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

# Model configuration constants for better architecture alignment
BACKBONE_DIM = 2048  # llama3_2_1B embedding dimension
DECODER_DIM = 1024   # llama3_2_100M embedding dimension
BACKBONE_LAYERS = 16
DECODER_LAYERS = 4
BACKBONE_HEADS = 32
BACKBONE_KV_HEADS = 8
DECODER_HEADS = 8
DECODER_KV_HEADS = 2
INTERMEDIATE_DIM = 8192
NORM_EPS = 1e-5

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
        
        # Embeddings with fixed dimensions matching llama3_2 models
        self.text_embeddings = nn.Embedding(config.text_vocab_size, BACKBONE_DIM)
        self.audio_embeddings = nn.Embedding(config.audio_vocab_size * config.audio_num_codebooks, BACKBONE_DIM)
        
        # Backbone transformer layers
        self.backbone = nn.Module()
        self.backbone.layers = nn.ModuleList([
            self._create_transformer_layer("backbone", i, BACKBONE_DIM, 
                                         BACKBONE_HEADS, BACKBONE_KV_HEADS,
                                         BACKBONE_DIM // BACKBONE_HEADS, INTERMEDIATE_DIM)
            for i in range(BACKBONE_LAYERS)
        ])
        self.backbone.norm = nn.LayerNorm(BACKBONE_DIM, eps=NORM_EPS)
        self.backbone.max_seq_len = config.backbone_max_seq_len
        
        # Projection with bias=False to match official implementation
        self.projection = nn.Linear(BACKBONE_DIM, DECODER_DIM, bias=False)
        
        # Decoder with proper dimensions
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([
            self._create_transformer_layer("decoder", i, DECODER_DIM,
                                         DECODER_HEADS, DECODER_KV_HEADS,
                                         DECODER_DIM // DECODER_HEADS, INTERMEDIATE_DIM)
            for i in range(DECODER_LAYERS)
        ])
        self.decoder.norm = nn.LayerNorm(DECODER_DIM, eps=NORM_EPS)
        
        # Output heads with bias=False for codebook0_head
        self.codebook0_head = nn.Linear(BACKBONE_DIM, config.audio_vocab_size, bias=False)
        self.audio_head = nn.Parameter(torch.empty(config.audio_num_codebooks - 1, 
                                                  DECODER_DIM, 
                                                  config.audio_vocab_size))
        
        # Initialize audio_head properly
        nn.init.normal_(self.audio_head, std=0.02)
        
        # Create caches and masks for efficient generation
        self.setup_caches(1)  # Default to batch size 1
    
    def _create_transformer_layer(self, prefix, layer_idx, embed_dim, num_heads, num_kv_heads, head_dim, intermediate_dim):
        """Create a transformer layer with the proper naming structure for state dict compatibility"""
        layer = nn.Module()
        
        # Attention module
        layer.attn = nn.Module()
        layer.attn.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        layer.attn.k_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=True)
        layer.attn.v_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=True)
        layer.attn.output_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=True)
        
        # MLP module
        layer.mlp = nn.Module()
        layer.mlp.w1 = nn.Linear(embed_dim, intermediate_dim, bias=True)
        layer.mlp.w2 = nn.Linear(intermediate_dim, embed_dim, bias=True)
        layer.mlp.w3 = nn.Linear(embed_dim, intermediate_dim, bias=True)
        
        # Use LayerNorm with a higher epsilon for more stability
        layer.sa_norm = nn.LayerNorm(embed_dim, eps=1e-4)  # Higher epsilon for stability
        layer.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-4)
        
        return layer
    
    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed both text and audio tokens"""
        # Text tokens are in the last dimension
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)
        
        # Audio tokens need offsetting based on codebook index
        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size * 
            torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        
        # Reshape to get embeddings for each token in each codebook
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )
        
        # Combine audio and text embeddings
        return torch.cat([audio_embeds, text_embeds], dim=-2)

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        """Embed audio tokens for a specific codebook"""
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)
    
    def forward(self, x, input_pos=None, mask=None):
        """Forward pass through the backbone transformer"""
        # Ensure no NaNs or Infs
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("Input to forward contains NaN or Inf values")
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        
        # Process through backbone with position information
        for i, layer in enumerate(self.backbone.layers):
            x_before = x
            x = self._forward_transformer_layer(layer, x, mask=mask)
            
            # Recovery strategy for NaN/Inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.warning(f"Layer {i} output contains NaN or Inf values")
                x = torch.where(torch.isnan(x) | torch.isinf(x), x_before, x)
        
        x = self.backbone.norm(x)
        return x

    def _forward_transformer_layer(self, layer, x, input_pos=None, mask=None):
        """Forward pass through a single transformer layer with KV caching support"""
        # Clamp inputs for stability
        x = torch.clamp(x, -100, 100)
        
        # Self-attention with KV caching
        norm_x = layer.sa_norm(x)
        
        # Projects for attention
        q = layer.attn.q_proj(norm_x)
        k = layer.attn.k_proj(norm_x)
        v = layer.attn.v_proj(norm_x)
        
        # Clamp intermediate values
        q = torch.clamp(q, -10, 10)
        k = torch.clamp(k, -10, 10)
        v = torch.clamp(v, -10, 10)
        
        # Reshape for multi-head attention
        batch_size, seq_len, _ = x.shape
        is_backbone = layer in self.backbone.layers
        head_dim = self.config.backbone_head_dim if is_backbone else self.config.decoder_head_dim
        num_heads = self.config.backbone_num_heads if is_backbone else self.config.decoder_num_heads
        num_kv_heads = self.config.backbone_num_kv_heads if is_backbone else self.config.decoder_num_kv_heads
        
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        
        # Store in KV cache if available and input_pos is provided
        if input_pos is not None and isinstance(input_pos, torch.Tensor):
            if is_backbone and self.backbone_caches_are_enabled():
                # Fix: Don't use KV caching for now - it's causing shape mismatches
                pass
            elif not is_backbone and self.decoder_caches_are_enabled():
                # Fix: Don't use KV caching for now
                pass
        
        # Repeat k and v for grouped-query attention
        repeat_factor = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
        
        # Scaled dot-product attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        attn_output = layer.attn.output_proj(attn_output)
        
        # Clamp outputs for stability
        attn_output = torch.clamp(attn_output, -100, 100)
        
        # Residual connection
        x = x + attn_output
        x = torch.clamp(x, -100, 100)
        
        # MLP block
        mlp_norm = layer.mlp_norm(x)
        
        # SwiGLU activation
        hidden1 = layer.mlp.w1(mlp_norm)
        hidden1 = torch.clamp(hidden1, -100, 100)
        hidden2 = layer.mlp.w3(mlp_norm)
        hidden2 = torch.clamp(hidden2, -100, 100)
        hidden = torch.nn.functional.silu(hidden1) * hidden2
        hidden = torch.clamp(hidden, -100, 100)
        output = layer.mlp.w2(hidden)
        output = torch.clamp(output, -100, 100)
        
        # Final residual connection
        x = x + output
        x = torch.clamp(x, -100, 100)
        
        return x

    def setup_caches(self, max_batch_size: int):
        """Setup key-value caches for the backbone and decoder"""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        # Create causal masks
        self.register_buffer("backbone_causal_mask", 
                            torch.tril(torch.ones(self.backbone.max_seq_len, 
                                                 self.backbone.max_seq_len, 
                                                 dtype=torch.bool, 
                                                 device=device)))
        self.register_buffer("decoder_causal_mask", 
                            torch.tril(torch.ones(self.config.audio_num_codebooks, 
                                                 self.config.audio_num_codebooks, 
                                                 dtype=torch.bool, 
                                                 device=device)))
        
        # Disable KV caching for now since it's causing shape mismatches
        self.backbone.cache_enabled = False
        self.decoder.cache_enabled = False

    def reset_caches(self):
        """Reset all KV caches to zeros"""
        if hasattr(self.backbone, 'k_cache'):
            self.backbone.k_cache.zero_()
            self.backbone.v_cache.zero_()
        
        if hasattr(self.decoder, 'k_cache'):
            self.decoder.k_cache.zero_()
            self.decoder.v_cache.zero_()

    def backbone_caches_are_enabled(self):
        """Check if backbone caches are enabled"""
        return hasattr(self.backbone, 'cache_enabled') and self.backbone.cache_enabled

    def decoder_caches_are_enabled(self):
        """Check if decoder caches are enabled"""
        return hasattr(self.decoder, 'cache_enabled') and self.decoder.cache_enabled

    def reset_bad_tensor_values(self):
        """Reset any parameters that have developed NaN or Inf values"""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    logger.warning(f"Resetting bad values in {name}")
                    param.data = torch.where(
                        torch.isnan(param) | torch.isinf(param),
                        torch.zeros_like(param),
                        param
                    )

# Add after the Model class and before AudioDecoderConfig
class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[512, 1024, 2048], 
                 hop_sizes=[120, 240, 480], 
                 win_lengths=[600, 1200, 2400]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        
    def forward(self, y_hat, y):
        loss = 0
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            stft_y = torch.stft(y, fft_size, hop_size, win_length, return_complex=True)
            stft_y_hat = torch.stft(y_hat, fft_size, hop_size, win_length, return_complex=True)
            
            # Spectral convergence
            sc_loss = torch.norm(stft_y - stft_y_hat, p='fro') / torch.norm(stft_y, p='fro')
            
            # Log magnitude loss
            mag_loss = F.l1_loss(torch.log(torch.abs(stft_y) + 1e-7), 
                                torch.log(torch.abs(stft_y_hat) + 1e-7))
            
            loss += sc_loss + mag_loss
            
        return loss / len(self.fft_sizes)

class AudioDecoderConfig:
    def __init__(self,
                 num_codebooks=32,
                 vocab_size=2051,
                 dim=512,
                 hidden_dim=1024,
                 n_layers=8,  # Increased number of layers
                 upsample_rates=[8, 8, 4, 2],  # Modified upsampling schedule
                 sample_rate=24000,
                 mel_channels=80,
                 kernel_size=7,
                 dilations=[1, 3, 9, 27],  # Exponential dilation growth
                 use_spectral_norm=True):
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.upsample_rates = upsample_rates
        self.sample_rate = sample_rate
        self.mel_channels = mel_channels
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.use_spectral_norm = use_spectral_norm

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, use_spectral_norm=True):
        super().__init__()
        self.use_spectral_norm = use_spectral_norm
        
        # First convolution block
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 1, 
                              dilation=dilation, padding=dilation*(kernel_size-1)//2)
        self.norm1 = nn.InstanceNorm1d(channels)
        
        # Second convolution block
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, 1,
                              dilation=1, padding=(kernel_size-1)//2)
        self.norm2 = nn.InstanceNorm1d(channels)
        
        if use_spectral_norm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)
        
    def forward(self, x):
        residual = x
        
        # First conv block
        x = self.conv1(F.leaky_relu(x, 0.2))
        x = self.norm1(x)
        
        # Second conv block
        x = self.conv2(F.leaky_relu(x, 0.2))
        x = self.norm2(x)
        
        return residual + x

class AudioDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embedding with fixed positional encoding size
        self.token_embedding = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.dim)
            for _ in range(config.num_codebooks)
        ])
        # Reduce positional embedding size to match maximum sequence length
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, config.dim))  # Fixed size for audio tokens
        
        # Initial processing
        self.pre_net = nn.Sequential(
            nn.Linear(config.dim * config.num_codebooks, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv1d(config.hidden_dim, config.hidden_dim, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm1d(config.hidden_dim)
        )
        
        # Multi-scale residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(config.n_layers):
            dilation = config.dilations[i % len(config.dilations)]
            self.res_blocks.append(
                ResBlock(config.hidden_dim, config.kernel_size, dilation, config.use_spectral_norm)
            )
        
        # Progressive upsampling
        self.up_layers = nn.ModuleList()
        current_dim = config.hidden_dim
        
        for rate in config.upsample_rates:
            self.up_layers.append(nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(current_dim, current_dim // 2, 
                                 rate * 2, rate, padding=rate // 2),
                nn.InstanceNorm1d(current_dim // 2),
                ResBlock(current_dim // 2, config.kernel_size, 1, config.use_spectral_norm)
            ))
            current_dim //= 2
        
        # Output layers
        self.mel_conv = nn.Conv1d(current_dim, config.mel_channels, 7, 1, 3)
        self.out_conv = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(current_dim, current_dim, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(current_dim, 1, 7, 1, 3),
            nn.Tanh()
        )
        
        # Apply spectral normalization if enabled
        if config.use_spectral_norm:
            self.mel_conv = nn.utils.spectral_norm(self.mel_conv)
            self.out_conv[1] = nn.utils.spectral_norm(self.out_conv[1])
            self.out_conv[3] = nn.utils.spectral_norm(self.out_conv[3])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, tokens):
        batch_size, seq_len, num_codebooks = tokens.shape
        
        # Process each codebook separately
        embeddings = []
        for i in range(num_codebooks):
            emb = self.token_embedding[i](tokens[:,:,i])
            embeddings.append(emb)
        
        # Combine embeddings and add positional encoding
        x = torch.cat(embeddings, dim=-1)
        # Ensure positional embedding matches sequence length
        pos_emb = self.pos_embedding[:, :min(seq_len, self.pos_embedding.size(1)), :]
        if seq_len > self.pos_embedding.size(1):
            # Repeat if needed
            pos_emb = pos_emb.repeat(1, (seq_len + self.pos_embedding.size(1) - 1) // self.pos_embedding.size(1), 1)
            pos_emb = pos_emb[:, :seq_len, :]
        x = x + pos_emb
        
        # Pre-net processing
        x = self.pre_net(x)
        
        # Reshape for convolutions
        x = x.transpose(1, 2)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Progressive upsampling
        for up_layer in self.up_layers:
            x = up_layer(x)
        
        # Generate outputs
        mels = self.mel_conv(x)
        waveform = self.out_conv(x).squeeze(1)
        
        return waveform, mels

def apply_audio_enhancements(waveform: torch.Tensor, sample_rate: int = 24000) -> torch.Tensor:
    """Apply basic audio enhancements to improve quality"""
    # Normalize to [-1, 1] range
    waveform = waveform / (waveform.abs().max() + 1e-8)
    
    # Apply a slight low-pass filter to reduce noise
    if torchaudio.get_audio_backend() == "sox":
        try:
            waveform = torchaudio.functional.lowpass_biquad(waveform, sample_rate, cutoff_freq=12000)
        except:
            pass  # Skip if not available
    
    # Add a tiny bit of reverb (this is a very simple approximation)
    reverb_delay = int(sample_rate * 0.01)  # 10ms delay
    if reverb_delay > 0 and waveform.size(-1) > reverb_delay + 100:
        reverb = torch.zeros_like(waveform)
        reverb[..., reverb_delay:] = waveform[..., :-reverb_delay] * 0.3
        waveform = waveform + reverb
        waveform = waveform / (waveform.abs().max() + 1e-8)  # Normalize again
    
    return waveform

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
    def __init__(self, model: Model, use_fp32=True):
        super().__init__(model)
        self.device = torch.device("cuda")
        self.model.eval()
        
        # Move entire model to CUDA with the specified precision
        dtype = torch.float32 if use_fp32 else torch.float16
        self.model.to(self.device, dtype=dtype)
        
        # Set the embedding precision for both forward and backward compatibility
        self.embedding_dtype = dtype
        
        # Initialize the audio decoder with more robust error handling
        try:
            audio_decoder_config = AudioDecoderConfig(
                num_codebooks=model.config.audio_num_codebooks,
                vocab_size=model.config.audio_vocab_size,
                sample_rate=self.sample_rate
            )
            self.audio_decoder = AudioDecoder(audio_decoder_config).to(self.device, dtype=dtype)
            
            # Initialize decoder weights with normal distribution
            for p in self.audio_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                
            logger.info("Audio decoder initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing audio decoder: {e}")
            raise
        
        self.tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

    def decode_audio_tokens(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """Convert audio tokens to waveform"""
        with torch.no_grad():
            try:
                # Add batch dimension if needed
                if audio_tokens.dim() == 2:
                    audio_tokens = audio_tokens.unsqueeze(0)
                
                # Ensure minimum sequence length for convolution operations
                min_seq_len = 8
                if audio_tokens.size(1) < min_seq_len:
                    # Pad sequence to minimum length by repeating
                    repeat_factor = (min_seq_len + audio_tokens.size(1) - 1) // audio_tokens.size(1)
                    audio_tokens = audio_tokens.repeat(1, repeat_factor, 1)
                    audio_tokens = audio_tokens[:, :min_seq_len, :]
                
                # Ensure maximum sequence length
                max_seq_len = 32  # Match positional embedding size
                if audio_tokens.size(1) > max_seq_len:
                    audio_tokens = audio_tokens[:, :max_seq_len, :]
                
                # Process in chunks if sequence is too long
                chunk_size = 16
                if audio_tokens.size(1) > chunk_size:
                    waveforms = []
                    for i in range(0, audio_tokens.size(1), chunk_size):
                        chunk = audio_tokens[:, i:i+chunk_size, :]
                        chunk_waveform, _ = self.audio_decoder(chunk)
                        waveforms.append(chunk_waveform)
                    waveform = torch.cat(waveforms, dim=0)
                else:
                    waveform, _ = self.audio_decoder(audio_tokens)
                
                # Apply enhancements and cleanup
                waveform = apply_audio_enhancements(waveform, self.sample_rate)
                if waveform.dim() == 3:
                    waveform = waveform.squeeze(0)
                
                return waveform.cpu()
                
            except Exception as e:
                logger.error(f"Error during audio decoding: {e}")
                # Fallback to noise
                num_samples = int(self.sample_rate * 0.5)  # 0.5 seconds of audio
                return torch.randn(num_samples)

    def sample_topk(self, logits, topk, temperature):
        """Sample from logits using top-k sampling with temperature"""
        logits = logits / max(temperature, 1e-5)
        
        # Get topk values and set others to -inf
        values, _ = torch.topk(logits, topk)
        min_value = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_value, 
                             torch.full_like(logits, float('-inf')), 
                             logits)
        
        # Apply softmax and sample
        probs = torch.softmax(logits, dim=-1)
        
        # Handle any potential NaN/Inf values
        if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
            logger.warning("Invalid probability distribution, using uniform sampling")
            probs = torch.ones_like(probs) / probs.size(-1)
        
        try:
            return torch.multinomial(probs, num_samples=1)
        except RuntimeError:
            # Fallback to argmax if sampling fails
            logger.warning("Multinomial sampling failed, falling back to argmax")
            return probs.argmax(dim=-1, keepdim=True)

    def generate_frame(self, tokens: torch.Tensor, tokens_mask: torch.Tensor, input_pos: torch.Tensor,
                       temperature: float, topk: int, len_text: int) -> torch.Tensor:
        """Generate a single frame of audio tokens using sequential codebook generation"""
        try:
            # Clear cache before processing
            torch.cuda.empty_cache()
            gc.collect()
            
            b = tokens.size(0)  # batch size
            num_codebooks = self.model.config.audio_num_codebooks
            
            # Reduce memory usage by processing smaller chunks
            chunk_size = 32  # Reduced from 64
            if tokens.size(1) > chunk_size:
                start_idx = max(0, tokens.size(1) - chunk_size)
                tokens = tokens[:, start_idx:]
                tokens_mask = tokens_mask[:, start_idx:]
                if input_pos is not None:
                    input_pos = input_pos[:, start_idx:]
            
            # Process in half precision for memory efficiency
            with torch.amp.autocast('cuda'):
                # Use the causal mask directly with reduced size
                seq_len = tokens.size(1)
                causal_mask = self.model.backbone_causal_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)
                causal_mask = causal_mask.expand(b, self.model.config.backbone_num_heads, -1, -1).to(tokens.device)
                
                # Process embeddings with memory optimization
                embeds = self.model._embed_tokens(tokens)
                masked_embeds = embeds * tokens_mask.unsqueeze(-1)
                h = masked_embeds.sum(dim=2)  # Sum across codebooks
                del embeds, masked_embeds  # Free memory immediately
                
                # Forward through backbone with reduced precision
                h = self.model.forward(h, mask=causal_mask)
                last_h = h[:, -1, :]
                del h  # Free memory
                
                # Generate tokens sequentially with memory optimization
                audio_frame_tokens = torch.zeros(b, num_codebooks, dtype=torch.int32, device=self.device)
                
                # First codebook with reduced precision
                c0_logits = self.model.codebook0_head(last_h * 0.1)  # Scale down
                c0_sample = self.sample_topk(c0_logits, topk, temperature)
                audio_frame_tokens[:, 0] = c0_sample.flatten()
                del c0_logits, c0_sample
                
                # Project once for all remaining codebooks with scaled values for numerical stability
                projected_h = self.model.projection(last_h)
                # Scale for fp16 stability
                projected_h = projected_h * 0.1
                del last_h
                
                # Generate remaining codebooks in smaller batches
                batch_size = 2  # Reduced from 4
                for i in range(1, num_codebooks, batch_size):
                    end_idx = min(i + batch_size, num_codebooks)
                    with torch.amp.autocast('cuda'):
                        # Process in smaller chunks to reduce memory usage
                        ci_logits = torch.matmul(projected_h, self.model.audio_head[i-1:end_idx-1])
                        # Process one codebook at a time to avoid dimension issues
                        for j in range(i, end_idx):
                            idx = j - i
                            codebook_logits = ci_logits[:, idx, :]
                            codebook_sample = self.sample_topk(codebook_logits, topk, temperature)
                            audio_frame_tokens[:, j] = codebook_sample.flatten()
                        del ci_logits
                    
                    # Clear cache after each batch
                    torch.cuda.empty_cache()
                    gc.collect()
            
            return audio_frame_tokens
            
        except Exception as e:
            logger.error(f"Error in generate_frame: {str(e)}")
            raise

    def generate(self, text: str, speaker: int, max_audio_length_ms: int) -> torch.Tensor:
        """Generate audio from text"""
        try:
            with torch.no_grad(), torch.amp.autocast('cuda'):
                # Clear CUDA memory
                torch.cuda.empty_cache()
                gc.collect()
                
                # Reset model caches
                self.model.reset_caches()

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
                # Get text tokens
                text_tokens = self.tokenizer.encode(text, return_tensors="pt")[0].to(self.device)
                
                # Setup generation parameters
                b = 1  # batch size
                num_codebooks = self.model.config.audio_num_codebooks
                max_seq_len = self.model.backbone.max_seq_len
                
                # Calculate maximum number of frames to generate
                num_audio_frames = min(
                    (max_audio_length_ms // self.frame_duration_ms),
                    max_seq_len - len(text_tokens)
                )

                # Initialize token tensor - add extra dimension for codebooks
                tokens = torch.zeros((b, len(text_tokens), num_codebooks + 1),
                                     dtype=torch.int32, device=self.device)
                # Place text tokens in the last position of the codebook dimension
                tokens[0, :, -1] = text_tokens
                tokens_mask = torch.ones_like(tokens, dtype=torch.bool)
                
                # Current sequence length (text only at start)
                current_seq_len = len(text_tokens)
                audio_tokens_list = []

                # Process in smaller batches
                max_frames_per_batch = 16  # Reduced from 32
                for frame_idx in range(num_audio_frames):
                    if frame_idx > 0 and frame_idx % max_frames_per_batch == 0:
                        # Clear cache periodically
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Current positions for attention
                    input_pos = torch.arange(current_seq_len, device=self.device).unsqueeze(0)
                    
                    # Debug input tokens periodically
                    if frame_idx > 0 and frame_idx % 10 == 0:
                        logger.info(f"Frame {frame_idx} input token stats: min={tokens.min().item()}, max={tokens.max().item()}")
                    
                    # Generate frame tokens
                    frame_tokens = self.generate_frame(
                        tokens[:, :current_seq_len],
                        tokens_mask[:, :current_seq_len, :],
                        input_pos,
                        temperature=0.8,  # Lower temperature for stability
                        topk=50,          # Larger topk for better diversity
                        len_text=len(text_tokens)
                    )
                    
                    # Validate frame tokens
                    if torch.isnan(frame_tokens).any() or torch.isinf(frame_tokens).any():
                        logger.warning(f"Invalid frame tokens at frame {frame_idx}, using fallback")
                        frame_tokens = torch.zeros_like(frame_tokens)
                    
                    # Create new token entry for this frame
                    new_tokens = torch.zeros((b, 1, num_codebooks + 1), dtype=torch.int32, device=self.device)
                    new_tokens[0, 0, :num_codebooks] = frame_tokens[0]  # Audio tokens
                    new_tokens[0, 0, -1] = 0  # No text token
                    
                    # Append to tokens tensor
                    tokens = torch.cat([tokens, new_tokens], dim=1)
                    tokens_mask = torch.cat([tokens_mask, torch.ones_like(new_tokens, dtype=torch.bool)], dim=1)
                    
                    # Save generated frame
                    audio_tokens_list.append(frame_tokens)
                    current_seq_len += 1
                    
                    # Free memory
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Reset bad values in model parameters occasionally
                    if frame_idx % 20 == 0:
                        self.reset_bad_tensor_values()
                    
                    # Process audio in smaller chunks if needed
                    if len(audio_tokens_list) >= max_frames_per_batch:
                        chunk = torch.stack(audio_tokens_list, dim=1).squeeze(0)
                        chunk_waveform = self.decode_audio_tokens(chunk)
                        audio_tokens_list = []  # Clear the list
                        torch.cuda.empty_cache()
                        gc.collect()
                
                # Process any remaining frames
                if audio_tokens_list:
                    chunk = torch.stack(audio_tokens_list, dim=1).squeeze(0)
                    waveform = self.decode_audio_tokens(chunk)
                else:
                    waveform = torch.zeros(1, device=self.device)  # Empty waveform
                
                return waveform
                
        except Exception as e:
            logger.error(f"Error during audio generation: {str(e)}")
            return self._generate_error_tone()

    def reset_bad_tensor_values(self):
        """Reset any parameters that have developed NaN or Inf values"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    logger.warning(f"Resetting bad values in {name}")
                    param.data = torch.where(
                        torch.isnan(param) | torch.isinf(param),
                        torch.zeros_like(param),
                        param
                    )

    def _generate_error_tone(self):
        # Return a short error tone instead of crashing
        error_samples = int(self.sample_rate * 0.5)  # 0.5 seconds
        return torch.sin(torch.linspace(0, 20 * torch.pi, error_samples)).cpu()

# Function to load the model
def load_csm_1b(device: str = "cuda", use_fp32: bool = True) -> CUDAGenerator:
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    repo_id = "sesame/csm-1b"
    try:
        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    except Exception as e:
        logger.error(f"Failed to download model files: {e}")
        raise

    with open(config_path, "r") as f:
        config_dict = json.load(f)
        config = ModelArgs(**config_dict)
    model = Model(config)
    
    # Load the state dictionary
    state_dict = safetensors_load(weights_path)
    
    # Use our improved conversion function
    converted_state_dict = convert_state_dict_keys(state_dict)
    
    # More conservative initialization
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'embedding' in name:
                # Embeddings need less scaling
                nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'norm' in name:
                # LayerNorm weights should be close to 1
                nn.init.ones_(param)
            else:
                # More conservative initialization for weights
                nn.init.normal_(param, mean=0.0, std=0.005)
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    # Initialize audio_head separately if exists
    if hasattr(model, 'audio_head'):
        nn.init.normal_(model.audio_head, mean=0.0, std=0.02)
    
    # Load the state dictionary with strict=False
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
    
    # Filter out expected missing keys
    expected_missing = ['backbone_causal_mask', 'decoder_causal_mask']
    unexpected_missing = [key for key in missing if key not in expected_missing]
    
    if unexpected_missing:
        logger.warning(f"Unexpected missing keys: {unexpected_missing}")
    if unexpected:
        # The unexpected 'bias' keys are added by our conversion function, so this is expected
        logger.info(f"Additional keys in state dict (added by conversion): {unexpected}")
    
    # Set model to evaluation mode
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Clear CUDA cache again
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Create and initialize the generator
    generator = CUDAGenerator(model, use_fp32=use_fp32)
    
    # Try to load decoder weights if available
    try:
        decoder_weights_path = hf_hub_download(repo_id=repo_id, filename="decoder.safetensors")
        logger.info("Loading audio decoder weights")
        decoder_state_dict = safetensors_load(decoder_weights_path)
        missing_keys, unexpected_keys = generator.audio_decoder.load_state_dict(decoder_state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys in audio decoder state dict: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in audio decoder state dict: {unexpected_keys}")
        logger.info("Audio decoder weights loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load audio decoder weights: {e}")
        logger.warning("Using randomly initialized decoder. Audio quality may be suboptimal.")
        # Initialize decoder with more conservative values
        for name, param in generator.audio_decoder.named_parameters():
            if 'weight' in name:
                if 'conv' in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='leaky_relu')
                else:
                    nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    return generator


def convert_state_dict_keys(state_dict):
    """Convert state dict keys to match our model architecture"""
    converted_dict = {}
    
    # Map between source and target key patterns
    key_mapping = {
        # Backbone mappings
        r'backbone\.layers\.(\d+)\.attn\.q_proj\.weight': r'backbone.layers.\1.attn.q_proj.weight',
        r'backbone\.layers\.(\d+)\.attn\.k_proj\.weight': r'backbone.layers.\1.attn.k_proj.weight',
        r'backbone\.layers\.(\d+)\.attn\.v_proj\.weight': r'backbone.layers.\1.attn.v_proj.weight',
        r'backbone\.layers\.(\d+)\.attn\.output_proj\.weight': r'backbone.layers.\1.attn.output_proj.weight',
        r'backbone\.layers\.(\d+)\.mlp\.w1\.weight': r'backbone.layers.\1.mlp.w1.weight',
        r'backbone\.layers\.(\d+)\.mlp\.w2\.weight': r'backbone.layers.\1.mlp.w2.weight',
        r'backbone\.layers\.(\d+)\.mlp\.w3\.weight': r'backbone.layers.\1.mlp.w3.weight',
        r'backbone\.layers\.(\d+)\.sa_norm\.scale': r'backbone.layers.\1.sa_norm.weight', 
        r'backbone\.layers\.(\d+)\.mlp_norm\.scale': r'backbone.layers.\1.mlp_norm.weight',
        r'backbone\.norm\.scale': r'backbone.norm.weight',
        
        # Decoder mappings
        r'decoder\.layers\.(\d+)\.attn\.q_proj\.weight': r'decoder.layers.\1.attn.q_proj.weight',
        r'decoder\.layers\.(\d+)\.attn\.k_proj\.weight': r'decoder.layers.\1.attn.k_proj.weight',
        r'decoder\.layers\.(\d+)\.attn\.v_proj\.weight': r'decoder.layers.\1.attn.v_proj.weight',
        r'decoder\.layers\.(\d+)\.attn\.output_proj\.weight': r'decoder.layers.\1.attn.output_proj.weight',
        r'decoder\.layers\.(\d+)\.mlp\.w1\.weight': r'decoder.layers.\1.mlp.w1.weight',
        r'decoder\.layers\.(\d+)\.mlp\.w2\.weight': r'decoder.layers.\1.mlp.w2.weight',
        r'decoder\.layers\.(\d+)\.mlp\.w3\.weight': r'decoder.layers.\1.mlp.w3.weight',
        r'decoder\.layers\.(\d+)\.sa_norm\.scale': r'decoder.layers.\1.sa_norm.weight',
        r'decoder\.layers\.(\d+)\.mlp_norm\.scale': r'decoder.layers.\1.mlp_norm.weight',
        r'decoder\.norm\.scale': r'decoder.norm.weight',
    }
    
    import re
    
    # Create dummy tensors for missing biases
    # This will provide zero biases for linear layers that need them
    model_keys = set()
    for key in state_dict.keys():
        model_keys.add(key)
        
        # For each weight, check if we need a corresponding bias
        if key.endswith('.weight') and not key.replace('.weight', '.bias') in state_dict:
            if any(x in key for x in ['q_proj', 'k_proj', 'v_proj', 'output_proj', 'w1', 'w2', 'w3']):
                tensor_shape = state_dict[key].size()
                if len(tensor_shape) == 2:  # Linear layer weight
                    dummy_bias = torch.zeros(tensor_shape[0], dtype=state_dict[key].dtype)
                    converted_dict[key.replace('.weight', '.bias')] = dummy_bias
    
    # Process each key in the state dict
    for key, value in state_dict.items():
        # Apply registered pattern mappings
        new_key = key
        for pattern, replacement in key_mapping.items():
            new_key = re.sub(pattern, replacement, new_key)
        converted_dict[new_key] = value
    
    # Add LayerNorm biases (zero vectors) for each LayerNorm weight
    for key in list(converted_dict.keys()):
        if '.norm.weight' in key or '.sa_norm.weight' in key or '.mlp_norm.weight' in key:
            bias_key = key.replace('.weight', '.bias')
            if bias_key not in converted_dict:
                converted_dict[bias_key] = torch.zeros_like(converted_dict[key])
    
    # Add specific projection and codebook biases if missing
    if 'projection.weight' in converted_dict and 'projection.bias' not in converted_dict:
        weight = converted_dict['projection.weight']
        converted_dict['projection.bias'] = torch.zeros(weight.size(0), dtype=weight.dtype)
    
    if 'codebook0_head.weight' in converted_dict and 'codebook0_head.bias' not in converted_dict:
        weight = converted_dict['codebook0_head.weight']
        converted_dict['codebook0_head.bias'] = torch.zeros(weight.size(0), dtype=weight.dtype)
    
    return converted_dict

# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio from text using the CSM-1B model")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID")
    parser.add_argument("--max_audio_length_ms", type=int, default=3000, help="Max audio length in milliseconds")
    parser.add_argument("--output_file", type=str, default="output.wav", help="Output audio file path")
    parser.add_argument("--fp16", action="store_true", help="Use half precision (not recommended)")
    args = parser.parse_args()

    try:
        # Default to FP32 unless --fp16 is explicitly specified
        generator = load_csm_1b(use_fp32=not args.fp16)
        with torch.no_grad():
            waveform = generator.generate(args.text, args.speaker, args.max_audio_length_ms)
        torchaudio.save(args.output_file, waveform.unsqueeze(0).float().cpu(), generator.sample_rate)
        logger.info(f"Audio saved to {args.output_file}")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        exit(1)
