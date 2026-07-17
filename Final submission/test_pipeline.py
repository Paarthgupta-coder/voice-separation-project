import os
import subprocess
import torch
import torchaudio
import math

def create_dummy_wav(path, sample_rate=8000, duration_sec=3):
    """Creates a 3-second dummy WAV file with mixed sine waves."""
    t = torch.linspace(0, duration_sec, int(sample_rate * duration_sec))
    # Mix 3 sine waves at different frequencies
    f1, f2, f3 = 400, 600, 800
    wave1 = torch.sin(2 * math.pi * f1 * t)
    wave2 = torch.sin(2 * math.pi * f2 * t)
    wave3 = torch.sin(2 * math.pi * f3 * t)
    
    mixed = (wave1 + wave2 + wave3) / 3.0
    mixed = mixed.unsqueeze(0)  # Shape: (1, time)
    
    torchaudio.save(path, mixed, sample_rate)

def test_pipeline():
    test_input = "dummy_test_mix.wav"
    test_outdir = "test_outputs"
    
    print("Creating dummy input file...")
    create_dummy_wav(test_input)
    
    print("Running pipeline...")
    # NOTE: Since this requires downloading large models and having HF_TOKEN set,
    # this smoke test will fail if dependencies/tokens aren't set up.
    # To run this successfully locally, set HF_TOKEN and ensure space for models.
    
    if "HF_TOKEN" not in os.environ:
        print("\n[SKIP] HF_TOKEN is not set. Skipping the actual pipeline execution smoke test.")
        print("Please set the HF_TOKEN environment variable and run this test again.")
        if os.path.exists(test_input):
            os.remove(test_input)
        return

    # Call main script via subprocess
    cmd = [
        "python", "main.py",
        "--input", test_input,
        "--outdir", test_outdir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
        
    assert result.returncode == 0, "Pipeline failed with non-zero exit code"
    
    # Check outputs
    assert os.path.exists(test_outdir), "Output directory not created"
    output_files = [f for f in os.listdir(test_outdir) if f.endswith(".wav")]
    assert len(output_files) >= 1, "No output WAV files generated"
    
    print(f"Smoke test passed! Generated {len(output_files)} files.")
    
    # Cleanup
    if os.path.exists(test_input):
        os.remove(test_input)

if __name__ == "__main__":
    test_pipeline()
