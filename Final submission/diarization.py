import torch
import config
from pyannote.audio import Pipeline

class Diarizer:
    def __init__(self):
        if not config.HF_TOKEN:
            raise ValueError(
                "HF_TOKEN environment variable is not set! "
                "Pyannote diarization requires a HuggingFace access token."
            )
        print("Loading Pyannote Diarization pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            config.PYANNOTE_MODEL,
            use_auth_token=config.HF_TOKEN
        )

    def estimate_speaker_count(self, waveform, sample_rate=config.SEPFORMER_SAMPLE_RATE):
        """
        Estimates the number of speakers in the given waveform tensor.
        waveform: shape (1, time)
        Returns an integer count of unique speakers.
        """
        # Guard against very short audio causing Pyannote to misfire/crash
        duration_sec = waveform.shape[1] / sample_rate
        if duration_sec < config.PYANNOTE_MIN_DURATION_SEC:
            print(f"  [Diarizer] Audio duration ({duration_sec:.2f}s) is below threshold "
                  f"({config.PYANNOTE_MIN_DURATION_SEC}s). Assuming 1 speaker.")
            return 1
            
        # pyannote expects dictionary with waveform and sample_rate
        with torch.no_grad():
            diarization = self.pipeline({"waveform": waveform, "sample_rate": sample_rate})
            
        # extract unique speaker labels
        speakers = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            
        return len(speakers)
