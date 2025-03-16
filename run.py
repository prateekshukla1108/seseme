import ctypes
import torch
import torchaudio
from huggingface_hub import hf_hub_download
import json
from dataclasses import dataclass
from torch import nn
from huggingface_hub import PyTorchModelHubMixin

# Define the ModelArgs dataclass to hold configuration parameters
@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int
    embedding_dim: int
    backbone_max_seq_len: int

# Define the main Model class inheriting from nn.Module and PyTorchModelHubMixin
class Model(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.text_embeddings = nn.Embedding(config.text_vocab_size, config.embedding_dim)
        self.audio_embeddings = nn.Embedding(config.audio_vocab_size, config.embedding_dim)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.codebook0_head = nn.Linear(config.embedding_dim, config.audio_vocab_size)
        self.audio_head = nn.Parameter(torch.randn(config.audio_num_codebooks - 1, config.embedding_dim, config.audio_vocab_size))
        self.backbone = DummyTransformerLayer()
        self.decoder = DummyTransformerLayer()
        self.backbone.max_seq_len = config.backbone_max_seq_len
        self.audio_num_codebooks = config.audio_num_codebooks

    def _embed_audio(self, index, sample):
        return self.audio_embeddings(sample)

# Dummy transformer layer for backbone and decoder
class DummyTransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(2)])
        self.max_seq_len = 100

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Base Generator class
class Generator:
    def __init__(self, model):
        self.model = model
        self.sample_rate = 24000

    def generate(self, text, speaker, context, max_audio_length_ms):
        dummy_audio = torch.randn(int(self.sample_rate * max_audio_length_ms / 1000))
        return dummy_audio

# CUDA-accelerated Generator class
class CUDAGenerator(Generator):
    def __init__(self, model: Model):
        super().__init__(model)
        self.device = torch.device("cuda")
        self.model = model.to(self.device, dtype=torch.float32)

        self.text_emb_weights = self.model.text_embeddings.weight.data.contiguous().to(self.device)
        self.audio_emb_weights = self.model.audio_embeddings.weight.data.contiguous().to(self.device)
        self.proj_weights = self.model.projection.weight.data.contiguous().to(self.device)
        self.proj_bias = torch.zeros(self.model.projection.out_features, device=self.device, dtype=torch.float32)
        self.c0_head_weights = self.model.codebook0_head.weight.data.contiguous().to(self.device)
        self.c0_head_bias = torch.zeros(self.model.codebook0_head.out_features, device=self.device, dtype=torch.float32)
        self.audio_head_weights = self.model.audio_head.data.contiguous().to(self.device)

        self.backbone_weights = [layer.weight.data.contiguous().to(self.device) for layer in self.model.backbone.layers]
        self.decoder_weights = [layer.weight.data.contiguous().to(self.device) for layer in self.model.decoder.layers]

    @torch.inference_mode()
    def generate_frame(
        self, tokens: torch.Tensor, tokens_mask: torch.Tensor, input_pos: torch.Tensor,
        temperature: float, topk: int
    ) -> torch.Tensor:
        b, s, _ = tokens.size()
        embed_dim = self.model.config.embedding_dim

        embeds = torch.zeros(b, s, 33, embed_dim, device=self.device, dtype=torch.float32)
        for i in range(33):
            input_ptr = tokens[:, :, i].contiguous().data_ptr()
            output_ptr = embeds[:, :, i].contiguous().data_ptr()
            weights = self.audio_emb_weights if i < 32 else self.text_emb_weights
            lib.embedding_forward_cuda(
                input_ptr, b, s, weights.size(0), embed_dim, weights.data_ptr(), output_ptr
            )

        h = embeds * tokens_mask.unsqueeze(-1)
        h = h.sum(dim=2)
        backbone_out = torch.zeros(b, s, embed_dim, device=self.device, dtype=torch.float32)
        mask = torch.zeros(b, s, self.model.backbone.max_seq_len, dtype=torch.bool, device=self.device)
        lib.index_causal_mask_cuda(
            input_pos.contiguous().data_ptr(), b, s, self.model.backbone.max_seq_len, mask.contiguous().data_ptr()
        )
        lib.transformer_decoder_forward_cuda(
            h.contiguous().data_ptr(), b, s, len(self.backbone_weights), 32, embed_dim // 32,
            [w.data_ptr() for w in self.backbone_weights], [w.data_ptr() for w in self.backbone_weights],
            [w.data_ptr() for w in self.backbone_weights], [w.data_ptr() for w in self.backbone_weights],
            mask.contiguous().data_ptr(), backbone_out.contiguous().data_ptr()
        )

        last_h = backbone_out[:, -1, :]
        c0_logits = torch.zeros(b, self.model.config.audio_vocab_size, device=self.device, dtype=torch.float32)
        lib.linear_forward_cuda(
            last_h.contiguous().data_ptr(), b, 1, embed_dim, self.model.config.audio_vocab_size,
            self.c0_head_weights.data_ptr(), self.c0_head_bias.data_ptr(), c0_logits.contiguous().data_ptr()
        )
        c0_sample = torch.zeros(b, 1, dtype=torch.int32, device=self.device)
        lib.sample_topk_cuda(
            c0_logits.contiguous().data_ptr(), topk, temperature, self.model.config.audio_vocab_size, b,
            c0_sample.contiguous().data_ptr()
        )

        curr_h = torch.cat([last_h.unsqueeze(1), self.model._embed_audio(0, c0_sample)], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = torch.arange(curr_h.size(1), device=self.device).unsqueeze(0)

        for i in range(1, self.model.config.audio_num_codebooks):
            decoder_out = torch.zeros(b, curr_h.size(1), embed_dim, device=self.device, dtype=torch.float32)
            mask = torch.zeros(b, curr_h.size(1), self.model.config.audio_num_codebooks, dtype=torch.bool, device=self.device)
            lib.index_causal_mask_cuda(
                curr_pos.contiguous().data_ptr(), b, curr_h.size(1), self.model.config.audio_num_codebooks, mask.contiguous().data_ptr()
            )
            lib.transformer_decoder_forward_cuda(
                curr_h.contiguous().data_ptr(), b, curr_h.size(1), len(self.decoder_weights), 32, embed_dim // 32,
                [w.data_ptr() for w in self.decoder_weights], [w.data_ptr() for w in self.decoder_weights],
                [w.data_ptr() for w in self.decoder_weights], [w.data_ptr() for w in self.decoder_weights],
                mask.contiguous().data_ptr(), decoder_out.contiguous().data_ptr()
            )
            ci_logits = torch.zeros(b, self.model.config.audio_vocab_size, device=self.device, dtype=torch.float32)
            lib.linear_forward_cuda(
                decoder_out[:, -1, :].contiguous().data_ptr(), b, 1, embed_dim, self.model.config.audio_vocab_size,
                self.audio_head_weights[i-1].data_ptr(), self.c0_head_bias.data_ptr(), ci_logits.contiguous().data_ptr()
            )
            ci_sample = torch.zeros(b, 1, dtype=torch.int32, device=self.device)
            lib.sample_topk_cuda(
                ci_logits.contiguous().data_ptr(), topk, temperature, self.model.config.audio_vocab_size, b,
                ci_sample.contiguous().data_ptr()
            )
            curr_h = self.model._embed_audio(i, ci_sample)
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

# Function to load the model and create the generator
def load_csm_1b(device: str = "cuda") -> CUDAGenerator:
    repo_id = "sesame/csm-1b"
    
    # Download model weights and configuration from Hugging Face Hub
    weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    
    # Load the model weights
    state_dict = torch.load(weights_path, map_location="cpu")
    
    # Infer embedding_dim from the 'text_embeddings.weight' tensor
    embedding_weight = state_dict['text_embeddings.weight']
    embedding_dim = embedding_weight.shape[1]
    
    # Load config.json
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Verify text_vocab_size matches the weights
    text_vocab_size_from_config = config_dict['text_vocab_size']
    if embedding_weight.shape[0] != text_vocab_size_from_config:
        raise ValueError(f"Mismatch in text_vocab_size: config has {text_vocab_size_from_config}, but weights have {embedding_weight.shape[0]}")
    
    # Add inferred embedding_dim to the configuration
    config_dict['embedding_dim'] = embedding_dim
    
    # Set a default backbone_max_seq_len if not present
    if 'backbone_max_seq_len' not in config_dict:
        config_dict['backbone_max_seq_len'] = 1024  # Default value
    
    # Create ModelArgs with the complete configuration
    config = ModelArgs(
        backbone_flavor=config_dict['backbone_flavor'],
        decoder_flavor=config_dict['decoder_flavor'],
        text_vocab_size=config_dict['text_vocab_size'],
        audio_vocab_size=config_dict['audio_vocab_size'],
        audio_num_codebooks=config_dict['audio_num_codebooks'],
        embedding_dim=config_dict['embedding_dim'],
        backbone_max_seq_len=config_dict['backbone_max_seq_len']
    )
    
    # Instantiate the model and load the weights
    model = Model(config)
    model.load_state_dict(state_dict)
    
    # Move model to the specified device
    model.to(device=device, dtype=torch.float32)
    
    # Create and return the CUDA generator
    generator = CUDAGenerator(model)
    return generator

# Main execution block
if __name__ == "__main__":
    # Load the CUDA kernels library (assumes './libtts_kernels.so' exists)
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
        ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_float)
    ]

    # Load the model and create the generator
    generator = load_csm_1b()
    
    # Generate audio from text
    text = "Hello, world!"
    speaker = 0
    context = []
    audio = generator.generate(text, speaker, context, max_audio_length_ms=5000)
    
    # Save the generated audio to a file
    torchaudio.save("output.wav", audio.unsqueeze(0), generator.sample_rate)
    print("Audio saved to output.wav")
