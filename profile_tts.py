#!/usr/bin/env python3
import time
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import gc
import argparse

# Import both implementations
from generator import load_csm_1b, Segment as OriginalSegment
from seseme_tts_optimized import load_csm_1b_optimized, Segment as OptimizedSegment

class Profiler:
    def __init__(self, name: str):
        self.name = name
        self.timings = {}
        self.memory_usage = {}
        self.start_memory = 0
        self.start_time = 0
    
    def start(self, label: str):
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.start_memory = torch.cuda.memory_allocated()
    
    def end(self, label: str):
        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated()
        
        duration = end_time - self.start_time
        memory_used = end_memory - self.start_memory
        
        self.timings[label] = duration
        self.memory_usage[label] = memory_used
        
        return duration, memory_used
    
    def report(self):
        print(f"\n--- {self.name} Profiling Results ---")
        print(f"{'Operation':<30} {'Time (s)':<10} {'Memory (MB)':<15}")
        print("-" * 60)
        
        for label in self.timings:
            time_ms = self.timings[label]
            memory_mb = self.memory_usage[label] / (1024 * 1024)
            print(f"{label:<30} {time_ms:.6f}  {memory_mb:.2f}")
        
        return self.timings, self.memory_usage

def test_original_implementation(texts: List[str], profiler: Profiler) -> List[torch.Tensor]:
    print("Testing original implementation...")
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load model
    profiler.start("model_load")
    tts = load_csm_1b()
    _, _ = profiler.end("model_load")
    
    audio_samples = []
    for i, text in enumerate(texts):
        print(f"Generating audio for text {i+1}/{len(texts)}")
        
        profiler.start(f"generate_text_{i}")
        audio = tts.generate(
            text=text,
            speaker=0,
            context=[],
            temperature=0.7,
            topk=50
        )
        duration, mem = profiler.end(f"generate_text_{i}")
        print(f"Generated {audio.size(0)/tts.sample_rate:.2f}s audio in {duration:.2f}s")
        
        audio_samples.append(audio)
    
    return audio_samples

def test_optimized_implementation(texts: List[str], profiler: Profiler) -> List[torch.Tensor]:
    print("Testing optimized implementation...")
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load model
    profiler.start("model_load")
    tts = load_csm_1b_optimized()
    _, _ = profiler.end("model_load")
    
    audio_samples = []
    for i, text in enumerate(texts):
        print(f"Generating audio for text {i+1}/{len(texts)}")
        
        profiler.start(f"generate_text_{i}")
        audio = tts.generate(
            text=text,
            speaker=0,
            context=[],
            temperature=0.7,
            topk=50
        )
        duration, mem = profiler.end(f"generate_text_{i}")
        print(f"Generated {audio.size(0)/tts.sample_rate:.2f}s audio in {duration:.2f}s")
        
        audio_samples.append(audio)
    
    return audio_samples

def compare_results(original_audio: List[torch.Tensor], optimized_audio: List[torch.Tensor], 
                   original_timings: Dict[str, float], optimized_timings: Dict[str, float],
                   sample_rate: int, save_path: str = "comparison_results"):
    """Compare the results between original and optimized implementations"""
    print("\n=== Performance Comparison ===")
    
    # Calculate speedup for each operation
    speedups = {}
    for key in original_timings:
        if key in optimized_timings:
            speedups[key] = original_timings[key] / optimized_timings[key]
    
    # Print speedup table
    print(f"{'Operation':<30} {'Original (s)':<15} {'Optimized (s)':<15} {'Speedup':<10}")
    print("-" * 75)
    
    for key in speedups:
        print(f"{key:<30} {original_timings[key]:<15.6f} {optimized_timings[key]:<15.6f} {speedups[key]:<10.2f}x")
    
    # Calculate average speedup for generation operations
    gen_speedups = [v for k, v in speedups.items() if k.startswith("generate_text_")]
    if gen_speedups:
        avg_speedup = sum(gen_speedups) / len(gen_speedups)
        print(f"\nAverage generation speedup: {avg_speedup:.2f}x")
    
    # Save audio comparison
    import os
    os.makedirs(save_path, exist_ok=True)
    
    for i, (orig, opt) in enumerate(zip(original_audio, optimized_audio)):
        # Save audio files
        torchaudio.save(f"{save_path}/original_{i}.wav", orig.unsqueeze(0), sample_rate)
        torchaudio.save(f"{save_path}/optimized_{i}.wav", opt.unsqueeze(0), sample_rate)
        
        # Calculate waveform similarity
        min_len = min(orig.size(0), opt.size(0))
        orig_trimmed = orig[:min_len]
        opt_trimmed = opt[:min_len]
        
        # Calculate normalized cross-correlation
        correlation = torch.nn.functional.cosine_similarity(
            orig_trimmed.unsqueeze(0), opt_trimmed.unsqueeze(0)
        ).item()
        
        print(f"Audio sample {i} similarity: {correlation:.4f}")
        
        # Plot waveforms
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(orig_trimmed.cpu().numpy())
        plt.title(f"Original Implementation - Sample {i}")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        
        plt.subplot(2, 1, 2)
        plt.plot(opt_trimmed.cpu().numpy())
        plt.title(f"Optimized Implementation - Sample {i}")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/waveform_comparison_{i}.png")
        plt.close()
    
    # Plot speedup comparison
    ops = list(speedups.keys())
    speedup_values = list(speedups.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(ops, speedup_values)
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)
    plt.title("Speedup Comparison (Optimized vs Original)")
    plt.xlabel("Operation")
    plt.ylabel("Speedup Factor (higher is better)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{save_path}/speedup_comparison.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Profile and compare TTS implementations")
    parser.add_argument("--texts", nargs="+", default=[
        "Hello, this is a test of the optimized TTS engine.",
        "The quick brown fox jumps over the lazy dog.",
        "SesemeTTS is a highly optimized text-to-speech engine for maximum performance."
    ], help="List of texts to synthesize")
    parser.add_argument("--output", type=str, default="comparison_results", 
                        help="Output directory for results")
    parser.add_argument("--skip-original", action="store_true", 
                        help="Skip testing the original implementation")
    args = parser.parse_args()
    
    # Set up profilers
    original_profiler = Profiler("Original Implementation")
    optimized_profiler = Profiler("Optimized Implementation")
    
    # Test implementations
    original_audio = []
    if not args.skip_original:
        original_audio = test_original_implementation(args.texts, original_profiler)
        original_profiler.report()
    
    optimized_audio = test_optimized_implementation(args.texts, optimized_profiler)
    optimized_timings, _ = optimized_profiler.report()
    
    # Compare results if both implementations were tested
    if not args.skip_original:
        # Get sample rate (assumes both implementations use the same sample rate)
        tts = load_csm_1b_optimized()
        sample_rate = tts.sample_rate
        
        original_timings, _ = original_profiler.report()
        compare_results(original_audio, optimized_audio, 
                       original_timings, optimized_timings,
                       sample_rate, args.output)

if __name__ == "__main__":
    main() 