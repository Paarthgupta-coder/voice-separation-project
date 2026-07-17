import os

# Separation Config
SEPFORMER_MODEL = "speechbrain/sepformer-wsj03mix"
SEPFORMER_SAVEDIR = "pretrained_models/sepformer-wsj03mix"
SEPFORMER_SAMPLE_RATE = 8000
MAX_RECURSION_DEPTH = 1

# Diarization Config
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
HF_TOKEN = os.environ.get("HF_TOKEN")
# Minimum duration (in seconds) of audio for pyannote to run.
# If a channel is shorter than this, we assume it's noise or not enough to hold >1 speaker.
PYANNOTE_MIN_DURATION_SEC = 2.0 

# Postprocessing Config
ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_SAVEDIR = "pretrained_models/spkrec-ecapa-voxceleb"
SILENCE_RMS_THRESHOLD_DB = -40.0
COSINE_SIMILARITY_THRESHOLD = 0.75
TARGET_NORMALIZATION_PEAK = 0.9
