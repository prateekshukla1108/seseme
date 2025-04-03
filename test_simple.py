#!/usr/bin/env python3
import torch
import torchaudio
import time
import argparse
from generator import load_csm_1b

def main():
    parser = argparse.ArgumentParser(description="Test the TTS engine")
    parser.add_argument("--text", type=str, default="Hello, this is a test of the TTS engine.", 
                      help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav", 
                       help="Output audio file")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--topk", type=int, default=50,
                      help="Top-k sampling parameter")
    parser.add_argument("--speaker", type=int, default=0,
                      help="Speaker ID")
    args = parser.parse_args()
    
    print("Loading TTS model...")
    start_time = time.time()
    
    # Load the model
    tts = load_csm_1b()
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Verify device
    print(f"Using device: {next(tts._model.parameters()).device}")
    
    # Generate audio
    print(f"Generating audio for text: '{args.text}'")
    start_time = time.time()
    
    audio = tts.generate(
        text=args.text,
        speaker=args.speaker,
        context=[],
        temperature=args.temperature,
        topk=args.topk
    )
    
    generation_time = time.time() - start_time
    audio_duration = audio.size(0) / tts.sample_rate
    
    print(f"Audio generated in {generation_time:.2f} seconds")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"Real-time factor: {generation_time / audio_duration:.4f}x")
    
    # Save audio to file
    # Move tensor to CPU before saving
    audio_cpu = audio.cpu()
    torchaudio.save(args.output, audio_cpu.unsqueeze(0), tts.sample_rate)
    print(f"Audio saved to {args.output}")
    
    # Print memory usage
    print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main() 