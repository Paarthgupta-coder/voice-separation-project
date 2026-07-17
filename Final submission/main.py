import argparse
import os
import torchaudio
import torch

import config
from separator import Separator
from diarization import Diarizer
from cascade import CascadePipeline
from postprocess import PostProcessor

def main():
    parser = argparse.ArgumentParser(description="Recursive Speech Separation Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input mixture .wav file")
    parser.add_argument("--outdir", type=str, default="./outputs", help="Directory to save separated speakers")
    args = parser.parse_args()

    input_path = args.input
    outdir = args.outdir
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    print(f"Loading input audio from: {input_path}")
    waveform, sample_rate = torchaudio.load(input_path)
    
    # 1. Resample if necessary to match SEPFORMER_SAMPLE_RATE (8kHz)
    if sample_rate != config.SEPFORMER_SAMPLE_RATE:
        print(f"Resampling from {sample_rate}Hz to {config.SEPFORMER_SAMPLE_RATE}Hz")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=config.SEPFORMER_SAMPLE_RATE)
        waveform = resampler(waveform)
        sample_rate = config.SEPFORMER_SAMPLE_RATE
        
    # Ensure it is mono
    if waveform.shape[0] > 1:
        print("Input is multi-channel. Averaging to mono.")
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Initialize Modules
    diarizer = Diarizer()
    separator = Separator()
    cascade = CascadePipeline(separator, diarizer)
    postprocessor = PostProcessor()
    
    # 2. Initial Speaker Count Estimate
    initial_estimate = diarizer.estimate_speaker_count(waveform, sample_rate)
    print(f"\n[INFO] Initial pyannote speaker count estimate: {initial_estimate} speakers")
    
    # 3. & 4. & 5. Base Separation and Recursive Re-split
    raw_channels_dict = cascade.process(waveform)
    
    print(f"\n[INFO] Finished separation. Raw candidate channels count: {len(raw_channels_dict)}")
    
    # 6. Silence Pruning
    pruned_dict, dropped = postprocessor.prune_silence(raw_channels_dict)
    print(f"\n[INFO] Silence Pruning:")
    for cid, rms in dropped:
        print(f"  - Dropped {cid} for being near-silent ({rms:.2f} dBFS)")
    print(f"  Surviving channels count: {len(pruned_dict)}")
    
    # 7. Speaker Deduplication
    print(f"\n[INFO] Speaker Deduplication:")
    dedup_dict, merges = postprocessor.deduplicate(pruned_dict)
    for c1, c2, sim in merges:
        print(f"  - Merged {c2} into {c1} (similarity score: {sim:.3f})")
    print(f"  Final channel count after pruning and dedup: {len(dedup_dict)}")
    
    # 8. Output Normalization and Saving
    print(f"\n[INFO] Saving Outputs to {outdir}:")
    for i, (cid, wav) in enumerate(dedup_dict.items()):
        norm_wav = postprocessor.normalize_loudness(wav)
        out_path = os.path.join(outdir, f"speaker_{i+1}.wav")
        # Save back as 8kHz, user can resample up later if they want
        torchaudio.save(out_path, norm_wav, sample_rate)
        print(f"  - Saved {out_path} (from provenance: {cid})")

    print("\n[INFO] Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
