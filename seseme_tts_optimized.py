import torch
import torchaudio
from typing import List, Tuple, Optional
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing
from moshi.models import loaders
from watermarking import load_watermarker, watermark, CSM_1B_GH_WATERMARK
import ctypes
from pathlib import Path

# Load the CUDA optimized library
_lib_path = Path(__file__).parent / "libseseme_tts_kernels.so"
_lib = ctypes.CDLL(str(_lib_path))

@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor

@dataclass
class ModelConfig:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int
    embed_dim: int
    decoder_dim: int
    max_seq_len: int
    num_layers: int
    num_heads: int
    num_kv_heads: int

def load_llama3_tokenizer():
    """
    Load the Llama 3 tokenizer with proper post-processing
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )
    return tokenizer

class OptimizedTTSModel:
    def __init__(self, model_config: ModelConfig, device: str = "cuda"):
        self.config = model_config
        self.device = torch.device(device)
        
        # Initialize CUDA context and allocate memory
        self._initialize_cuda_context()
        
        # Load tokenizers and other components
        self._text_tokenizer = load_llama3_tokenizer()
        
        # Load audio tokenizer (mimi)
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self._audio_tokenizer = loaders.get_mimi(mimi_weight, device=self.device)
        self._audio_tokenizer.set_num_codebooks(32)
        
        # Load watermarker
        self._watermarker = load_watermarker(device=device)
        
        self.sample_rate = self._audio_tokenizer.sample_rate
    
    def _initialize_cuda_context(self):
        """Initialize CUDA kernels and allocate memory for model parameters"""
        # Call into CUDA library to initialize context
        _lib.initializeModel(
            ctypes.c_int(self.config.embed_dim),
            ctypes.c_int(self.config.decoder_dim),
            ctypes.c_int(self.config.max_seq_len),
            ctypes.c_int(self.config.num_layers),
            ctypes.c_int(self.config.num_heads),
            ctypes.c_int(self.config.num_kv_heads),
            ctypes.c_int(self.config.text_vocab_size),
            ctypes.c_int(self.config.audio_vocab_size),
            ctypes.c_int(self.config.audio_num_codebooks)
        )
    
    def load_weights(self, model_path: str):
        """Load model weights from a checkpoint"""
        # Load weights and pass them to CUDA kernels
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract weights and convert to contiguous arrays for CUDA
        # This is a placeholder - actual implementation would extract all required weights
        text_embedding = checkpoint['text_embeddings.weight'].contiguous()
        audio_embedding = checkpoint['audio_embeddings.weight'].contiguous()
        
        # Pass weight pointers to CUDA kernels
        _lib.loadWeights(
            ctypes.c_void_p(text_embedding.data_ptr()),
            ctypes.c_void_p(audio_embedding.data_ptr())
        )
    
    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text segment with speaker information"""
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True
        
        return text_frame.to(self.device), text_frame_mask.to(self.device)
    
    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize audio segment"""
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True
        
        return audio_frame, audio_frame_mask
    
    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a complete segment (text + audio)"""
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
        
        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)
    
    def reset_caches(self):
        """Reset KV caches"""
        _lib.resetCaches()
    
    def setup_caches(self, max_batch_size: int):
        """Setup KV caches"""
        _lib.setupCaches(ctypes.c_int(max_batch_size))
    
    def generate_frame(self, tokens: torch.Tensor, tokens_mask: torch.Tensor, 
                      input_pos: torch.Tensor, temperature: float, topk: int) -> torch.Tensor:
        """Generate a frame using optimized CUDA kernels"""
        batch_size, seq_len, num_codebooks_plus_one = tokens.size()
        
        # Prepare inputs for CUDA kernel
        tokens_flat = tokens.contiguous().view(-1)
        tokens_mask_flat = tokens_mask.contiguous().view(-1)
        input_pos_flat = input_pos.contiguous().view(-1)
        
        # Allocate output tensor
        output = torch.zeros(batch_size, self.config.audio_num_codebooks, 
                            dtype=torch.int32, device=self.device)
        
        # Call CUDA kernel
        _lib.generateFrame(
            ctypes.c_void_p(tokens_flat.data_ptr()),
            ctypes.c_void_p(tokens_mask_flat.data_ptr()),
            ctypes.c_void_p(input_pos_flat.data_ptr()),
            ctypes.c_float(temperature),
            ctypes.c_int(topk),
            ctypes.c_int(batch_size),
            ctypes.c_int(seq_len),
            ctypes.c_void_p(output.data_ptr())
        )
        
        return output
    
    @torch.inference_mode()
    def generate(self, text: str, speaker: int, context: List[Segment], 
                max_audio_length_ms: float = 90_000, temperature: float = 0.9, 
                topk: int = 50) -> torch.Tensor:
        """Generate audio from text with full context"""
        self.reset_caches()
        
        max_audio_frames = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        
        # Process context segments
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)
        
        # Process new text segment
        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)
        
        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)
        
        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
        
        max_seq_len = 2048 - max_audio_frames
        if curr_tokens.size(1) >= max_seq_len:
            raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")
        
        # Generate frames
        for _ in range(max_audio_frames):
            sample = self.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # eos
            
            samples.append(sample)
            
            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1
        
        # Decode audio
        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)
        
        # Apply watermark
        audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
        audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)
        
        return audio

def load_csm_1b_optimized(device: str = "cuda") -> OptimizedTTSModel:
    """Load the CSM-1B model with optimized CUDA kernels"""
    config = ModelConfig(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-1B",
        text_vocab_size=128_256,
        audio_vocab_size=1024,
        audio_num_codebooks=32,
        embed_dim=2048,
        decoder_dim=2048,
        max_seq_len=2048,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8
    )
    
    model = OptimizedTTSModel(config, device=device)
    model.setup_caches(1)
    model.load_weights("sesame/csm-1b")
    
    return model 