# Voice Separation Project

This repository contains three approaches attempted for the same speech separation assignment: separating a single audio recording containing 3 to 5 overlapping speakers into individual speaker audio files, with speaker count unknown at test time.

- `Final submission/` : the approach used as the final deliverable, a recursive cascade built on a pretrained separator.
- `Try 2/` : an earlier attempt extending a pretrained MossFormer2 model with additional architecture and training.
- `Try 3/` : details to be added.

## Final submission

### What this is

The final submission separates speech using a pretrained SpeechBrain SepFormer checkpoint trained on WSJ0-3mix (`speechbrain/sepformer-wsj03mix`). This model is used entirely as is, with no fine tuning or training of any kind.

Since that checkpoint always produces exactly 3 output channels, the core idea here is a recursive cascade to reach 4 or 5 speakers without needing a differently sized model. After the initial 3 way split, each output channel is checked using speaker diarization (`pyannote/speaker-diarization-3.1`) to see whether it still contains more than one speaker. Any channel that still looks like a mixture is fed back into the same pretrained separator as a fresh input. This recursion is capped at one additional level.

The novelty of this approach is in reusing a single fixed 3-speaker model recursively, rather than needing a separate model per possible speaker count. This is supported by a diarization based recursion trigger with a minimum speaking duration filter to avoid over splitting on short leakage artifacts, silence pruning to drop near empty channels, speaker embedding based deduplication using ECAPA-TDNN to merge channels that turned out to be the same speaker split across branches, and a final output cap that keeps only the 5 loudest channels to respect the task's speaker count ceiling.

Training a dedicated model for 4 or 5 speakers was considered early on but was not pursued, since it would have required significantly more compute and time than was available within the project deadline and the Kaggle T4x2 environment. This is the main reason the recursive cascade design was chosen over building or training a differently sized model.

### Setup and how to run

This project was built and run on Kaggle notebooks with a GPU T4 x2 accelerator.

1. Create a HuggingFace token with read access at huggingface.co/settings/tokens, and accept the gated model terms on both `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` while logged in with that account.
2. In your Kaggle notebook, under Add-ons then Secrets, add a secret. The notebook reads it under the name `HF_TOKEN_2`. Either name your secret `HF_TOKEN_2` to match the notebook as is, or edit the line in the STEP 1 cell that calls `get_secret('HF_TOKEN_2')` to match whatever name you used.
3. Upload the contents of this folder (excluding the notebook) as a Kaggle Input dataset, and set the accelerator to GPU T4 x2 in Session options before running any cells.
4. Attach your own test audio file as a Kaggle Input dataset.
5. Run the notebook cells in order from the top.
6. In the STEP 8 cell, edit the `REAL_INPUT` variable to point at your own audio file's path. Run the `find /kaggle/input -iname "*.wav"` cell just above it first to get the exact path, since Kaggle's dataset mount paths do not always match the sidebar naming.
7. Output files are written to `/kaggle/working/outputs` as `speaker_1.wav` through `speaker_N.wav`. The output directory is cleared automatically at the start of each run.

The two lines that must be edited for a new user are the secret name in STEP 1 if a different secret name was used, and the `REAL_INPUT` path in STEP 8.

## Try 2

This was our first attempt at this problem, before the recursive cascade approach above.

This attempt used a pretrained MossFormer2 model trained on WSJ0-3Mix as a backbone, extended with two additional modules: an Adaptive Multi-Scale Temporal Module for capturing speech information at multiple temporal scales, and an Adaptive Gated Feature Fusion module for selectively combining features from different processing stages rather than simply concatenating them. Two auxiliary prediction heads were added alongside the separation output, a Speaker Count Head to estimate how many speakers are present in a mixture, and a Confidence Head to assign a reliability score to each separated output. All three objectives, separation, speaker count, and confidence, were trained jointly using a multi-task loss with uncertainty based weighting, so the network could balance the tasks automatically instead of using manually tuned loss weights.

Training started on a small custom three-speaker dataset to validate the architecture, then moved to the Libri3Mix dataset by swapping out the data loading pipeline while keeping the model unchanged. The encoder and decoder were kept frozen during this stage, with only the MaskNet and the newly added modules being trained, to reduce training cost while keeping the pretrained knowledge intact.

Training on the full Libri3Mix dataset turned out to be too time and compute intensive to complete within the available GPU runtime, so training was run on smaller subsets to validate the approach rather than to convergence. This is the main reason this direction was set aside in favor of the final submission.

## Try 3

Details to be added.
