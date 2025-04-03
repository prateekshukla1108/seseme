#!/usr/bin/env python3
import os
import sys
import argparse
import ctypes
import subprocess
import time
import tempfile
import platform
from pathlib import Path

class TTSError(Exception):
    """Custom exception for TTS-related errors"""
    pass

class SesameTTS:
    """Lightweight Python wrapper for the optimized C++ TTS engine"""
    
    def __init__(self, model_path=None, device="cuda"):
        """Initialize the TTS engine
        
        Args:
            model_path (str): Path to the model weights
            device (str): Device to run on ("cuda" or "cpu")
        """
        self.model_path = model_path or self._get_default_model_path()
        self.device = device
        
        # Check if CUDA is available if device is "cuda"
        if device == "cuda" and not self._is_cuda_available():
            print("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        
        # Ensure the CUDA library is built
        self._ensure_library_built()
        
        # Set environment variables
        if self.device == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Locate the TTS engine binary
        self.engine_path = self._get_engine_path()
        
        print(f"SesameTTS initialized with model: {self.model_path}")
        print(f"Using device: {self.device}")
    
    def _get_default_model_path(self):
        """Get the default model path"""
        # First check if there's a model in the current directory
        if os.path.exists("model.bin"):
            return os.path.abspath("model.bin")
        
        # Then check in the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "model.bin")
        if os.path.exists(model_path):
            return model_path
        
        # Finally, check in common locations
        common_locations = [
            os.path.expanduser("~/.sesame/models/"),
            "/usr/local/share/sesame/models/",
            "/usr/share/sesame/models/"
        ]
        
        for location in common_locations:
            model_path = os.path.join(location, "model.bin")
            if os.path.exists(model_path):
                return model_path
        
        # If no model found, return a placeholder and we'll error later
        return "model.bin"
    
    def _get_engine_path(self):
        """Get the path to the TTS engine binary"""
        # Check in the current directory
        if os.path.exists("./tts_engine"):
            return os.path.abspath("./tts_engine")
        
        # Check in the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(script_dir, "tts_engine")
        if os.path.exists(engine_path):
            return engine_path
        
        raise TTSError("Could not find TTS engine binary. Please build it first with 'make'.")
    
    def _is_cuda_available(self):
        """Check if CUDA is available"""
        try:
            # Try to get CUDA_HOME
            cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
            
            # If not set, try common locations
            if not cuda_home:
                common_locations = ["/usr/local/cuda", "/usr/cuda"]
                for loc in common_locations:
                    if os.path.exists(loc):
                        cuda_home = loc
                        break
            
            if not cuda_home:
                return False
            
            # Check for CUDA libraries
            lib_path = os.path.join(cuda_home, "lib64")
            return os.path.exists(lib_path) and any(
                os.path.exists(os.path.join(lib_path, f"lib{lib}.so")) 
                for lib in ["cuda", "cudart"]
            )
        except Exception:
            return False
    
    def _ensure_library_built(self):
        """Ensure the CUDA library is built"""
        # Check if the shared library exists
        if not os.path.exists("libtts_kernels.so"):
            print("CUDA library not found. Building it now...")
            
            # Check if we have a build script
            if os.path.exists("./build_cuda.sh"):
                subprocess.run(["bash", "./build_cuda.sh"], check=True)
            # Check if we have a makefile
            elif os.path.exists("./makefile") or os.path.exists("./Makefile"):
                subprocess.run(["make", "libtts_kernels.so"], check=True)
            else:
                raise TTSError("Could not build CUDA library. Please build it manually.")
    
    def synthesize(self, text, output_file=None, speaker_id=0, temperature=0.7, top_k=50):
        """Synthesize speech from text
        
        Args:
            text (str): The text to synthesize
            output_file (str): The output WAV file path (optional)
            speaker_id (int): Speaker ID to use (0-9)
            temperature (float): Sampling temperature (0.1-1.0)
            top_k (int): Top-k sampling parameter
            
        Returns:
            str: Path to the generated audio file
        """
        if not output_file:
            # Create a temporary file with a .wav extension
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
                output_file = temp.name
        
        # Prepare the command
        cmd = [
            self.engine_path,
            "--text", text,
            "--speaker", str(speaker_id),
            "--output", output_file,
            "--temperature", str(temperature),
            "--topk", str(top_k),
            "--model", self.model_path
        ]
        
        # Set library path
        env = os.environ.copy()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        env["LD_LIBRARY_PATH"] = f"{script_dir}:{env.get('LD_LIBRARY_PATH', '')}"
        
        # Run the command
        start_time = time.time()
        try:
            process = subprocess.run(
                cmd, 
                env=env,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            
            # Print engine output
            if process.stdout:
                print(process.stdout)
            
            end_time = time.time()
            print(f"Audio generated in {end_time - start_time:.2f} seconds")
            print(f"Saved to: {output_file}")
            
            return output_file
            
        except subprocess.CalledProcessError as e:
            print(f"Error running TTS engine: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr}")
            if os.path.exists(output_file):
                os.unlink(output_file)
            raise TTSError(f"Failed to synthesize speech: {e}")


def main():
    """Command-line interface for SesameTTS"""
    parser = argparse.ArgumentParser(description="Sesame TTS - Optimized Text-to-Speech")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("--output", "-o", help="Output WAV file")
    parser.add_argument("--speaker", "-s", type=int, default=0, help="Speaker ID (0-9)")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", "-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--model", "-m", help="Model path")
    parser.add_argument("--device", "-d", choices=["cuda", "cpu"], default="cuda", help="Device to use")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    try:
        # Initialize TTS engine
        tts = SesameTTS(model_path=args.model, device=args.device)
        
        if args.interactive:
            print("Sesame TTS Interactive Mode")
            print("Enter text to synthesize, or 'q' to quit")
            print("-" * 50)
            
            counter = 1
            while True:
                try:
                    text = input("> ")
                    if text.lower() in ("q", "quit", "exit"):
                        break
                    
                    if not text.strip():
                        continue
                    
                    output_file = args.output or f"output_{counter}.wav"
                    tts.synthesize(text, output_file, args.speaker, args.temperature, args.top_k)
                    counter += 1
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
            
            print("Exiting interactive mode")
        
        elif args.text:
            # Synthesize speech
            tts.synthesize(args.text, args.output, args.speaker, args.temperature, args.top_k)
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 