#!/usr/bin/env python3
import torch
import torchaudio
import time
import argparse
from seseme_tts_optimized import load_csm_1b_optimized

def main():
    parser = argparse.ArgumentParser(description="Test the optimized TTS engine")
    parser.add_argument("--text", type=str, default="Hello, this is a test of the optimized TTS engine.", 
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
    
    print("Loading optimized TTS model...")
    start_time = time.time()
    
    # Load the optimized model
    tts = load_csm_1b_optimized()
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Verify CUDA is being used
    print(f"Using device: {next(iter(tts._audio_tokenizer.parameters())).device}")
    
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
    torchaudio.save(args.output, audio.unsqueeze(0), tts.sample_rate)
    print(f"Audio saved to {args.output}")
    
    # Print memory usage
    print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main() 