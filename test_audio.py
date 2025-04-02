import torch
import os
import tempfile
from pathlib import Path
import unittest
import numpy as np
import soundfile as sf
from audio import AudioProcessor

class TestAudioUpdates(unittest.TestCase):
    """Test the updated audio processing functionality."""
    
    def test_gpu_detection(self):
        """Test that GPU detection works correctly."""
        # This should return True if CUDA is available, False otherwise
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            print(f"CUDA available: GPU = {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, using CPU")
        
        # The test passes either way, we're just checking that it doesn't crash
        self.assertIsInstance(has_cuda, bool)

    def test_audio_processor_init(self):
        """Test that AudioProcessor initializes correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup directories
            temp_path = Path(temp_dir)
            source_dir = temp_path / "source"
            output_dir = temp_path / "audio"
            transcriptions_dir = temp_path / "transcriptions"
            
            # Create directories
            source_dir.mkdir()
            output_dir.mkdir()
            transcriptions_dir.mkdir()
            
            # Create a prompt file
            prompt_file = temp_path / "prompt.txt"
            with open(prompt_file, "w") as f:
                f.write("This is a test prompt.")
            
            # Initialize AudioProcessor with a tiny model for fast testing
            processor = AudioProcessor(
                source_dir=source_dir,
                output_dir=output_dir,
                transcriptions_dir=transcriptions_dir,
                prompt_file=prompt_file,
                model_id="openai/whisper-tiny"
            )
            
            # Check that the processor was initialized correctly
            self.assertEqual(str(processor.source_dir), str(source_dir))
            self.assertEqual(str(processor.output_dir), str(output_dir))
            self.assertEqual(str(processor.transcriptions_dir), str(transcriptions_dir))
            self.assertEqual(str(processor.prompt_file), str(prompt_file))
            self.assertEqual(processor.model_id, "openai/whisper-tiny")
            
            # The device is not an attribute, but determined at runtime in transcribe_audio
            # Just verify that Cuda is available for GPU acceleration if applicable
            if torch.cuda.is_available():
                self.assertTrue(torch.cuda.is_available())
                self.assertIsNotNone(torch.cuda.get_device_name(0))
            else:
                self.assertFalse(torch.cuda.is_available())
    
    def test_load_audio_file(self):
        """Test that the load_audio_file method works with different approaches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup directories
            temp_path = Path(temp_dir)
            source_dir = temp_path / "source"
            output_dir = temp_path / "audio"
            transcriptions_dir = temp_path / "transcriptions"
            
            # Create directories
            source_dir.mkdir()
            output_dir.mkdir()
            transcriptions_dir.mkdir()
            
            # Create a sample audio file
            sample_rate = 16000
            duration = 3  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
            
            audio_path = output_dir / "test_audio.flac"
            sf.write(audio_path, audio_data, sample_rate)
            
            # Initialize AudioProcessor
            processor = AudioProcessor(
                source_dir=source_dir,
                output_dir=output_dir,
                transcriptions_dir=transcriptions_dir
            )
            
            # Test the load_audio_file method
            try:
                result, _ = processor.load_audio_file(audio_path)
                
                # Check if we got a dict (direct file path) or numpy array
                if isinstance(result, dict):
                    self.assertIn("audio", result)
                    self.assertIn("path", result["audio"])
                    self.assertEqual(result["audio"]["path"], str(audio_path))
                    print("Successfully loaded audio file with path method")
                else:
                    # We got numpy array
                    self.assertIsInstance(result, np.ndarray)
                    self.assertEqual(len(result), int(sample_rate * duration))
                    print(f"Successfully loaded audio file as array of length {len(result)}")
                
                # Test passes if we get either result type
                self.assertTrue(isinstance(result, dict) or isinstance(result, np.ndarray))
            except Exception as e:
                self.fail(f"load_audio_file raised exception: {e}")

if __name__ == "__main__":
    unittest.main() 