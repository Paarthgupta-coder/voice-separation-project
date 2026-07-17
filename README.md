# Voice Separation Project

Separating a single audio recording of 3 to 5 overlapping speakers into individual speaker audio files, with the speaker count unknown at test time.

This repository documents three attempts at the problem, in the order they were actually tried.

## Contents

- [Summary](#summary)
- [Final submission](#final-submission)
- [Try 2](#try-2)
- [Try 3](#try-3)

## Summary

| Folder | Approach | Training required | Outcome |
|---|---|---|---|
| `Final submission/` | Recursive cascade on a frozen pretrained 3-speaker SepFormer | None | Used as the final deliverable |
| `Try 2/` | Pretrained MossFormer2 extended with new modules and heads | Yes, on Libri3Mix | Set aside, full training too slow to complete |
| `Try 3/` | SepFormer checkpoint surgically widened from 3 to 4 speakers | Yes, one GPU day | Set aside, environment issues and incomplete convergence |

---

## Final submission

### What this is

Separation is done with a pretrained SpeechBrain SepFormer checkpoint trained on WSJ0-3mix (`speechbrain/sepformer-wsj03mix`). The model is used exactly as released, with no fine tuning or training of any kind.

Since this checkpoint always produces exactly 3 output channels, the core idea is a **recursive cascade** to reach 4 or 5 speakers without needing a differently sized model. After the initial 3 way split, each output channel is checked with speaker diarization (`pyannote/speaker-diarization-3.1`) to see whether it still contains more than one speaker. Any channel that still looks like a mixture is fed back into the same pretrained separator as a fresh input. Recursion is capped at one additional level.

**Supporting steps:**

- Diarization based recursion trigger, with a minimum speaking duration filter so short leakage artifacts are not mistaken for a second speaker
- Silence pruning to drop near empty channels
- Speaker embedding based deduplication using ECAPA-TDNN, to merge channels that turned out to be the same speaker split across different branches
- Output cap that keeps only the 5 loudest channels, matching the task's speaker count ceiling

The novelty here is reusing a single fixed 3-speaker model recursively, rather than needing a separate model per possible speaker count.

Training a dedicated model for 4 or 5 speakers was considered early on but was not pursued, since it would have required significantly more compute and time than was available within the project deadline and the Kaggle T4x2 environment. This is the main reason the recursive cascade design was chosen over building or training a differently sized model.

### Setup and how to run

Built and run on Kaggle notebooks with a GPU T4 x2 accelerator.

1. Create a HuggingFace token with read access at huggingface.co/settings/tokens, and accept the gated model terms on both `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`, while logged in with that account.
2. In your Kaggle notebook, under Add-ons then Secrets, add a secret. The notebook reads it under the name `HF_TOKEN_2`. Either name your secret `HF_TOKEN_2` to match the notebook as is, or edit the line in the STEP 1 cell that calls `get_secret('HF_TOKEN_2')` to match whatever name you used.
3. Upload the contents of this folder, excluding the notebook, as a Kaggle Input dataset, and set the accelerator to GPU T4 x2 in Session options before running any cells.
4. Attach your own test audio file as a Kaggle Input dataset.
5. Run the notebook cells in order from the top.
6. In the STEP 8 cell, edit the `REAL_INPUT` variable to point at your own audio file's path. Run the `find /kaggle/input -iname "*.wav"` cell just above it first to get the exact path, since Kaggle's dataset mount paths do not always match the sidebar naming.
7. Output files are written to `/kaggle/working/outputs` as `speaker_1.wav` through `speaker_N.wav`. The output directory is cleared automatically at the start of each run.

**Lines to edit for a new user:** the secret name in STEP 1, if a different secret name was used, and the `REAL_INPUT` path in STEP 8.

---

## Try 2

Our first attempt at this problem, tried before the recursive cascade above.

### Approach

Used a pretrained MossFormer2 model trained on WSJ0-3Mix as a backbone, extended with:

- An **Adaptive Multi-Scale Temporal Module**, for capturing speech information at multiple temporal scales
- An **Adaptive Gated Feature Fusion** module, for selectively combining features from different processing stages rather than simply concatenating them
- A **Speaker Count Head**, an auxiliary output estimating how many speakers are present in a mixture
- A **Confidence Head**, an auxiliary output assigning a reliability score to each separated output

All three objectives, separation, speaker count, and confidence, were trained jointly using a multi-task loss with uncertainty based weighting, so the network could balance the tasks automatically instead of using manually tuned loss weights.

### Training and why it was set aside

Training started on a small custom three-speaker dataset to validate the architecture, then moved to the Libri3Mix dataset by swapping out the data loading pipeline while keeping the model unchanged. The encoder and decoder were kept frozen, with only the MaskNet and the newly added modules trained, to reduce cost while keeping the pretrained knowledge intact.

Training on the full Libri3Mix dataset turned out to be too time and compute intensive to complete within the available GPU runtime, so training was run on smaller subsets to validate the approach rather than to convergence. This is the main reason this direction was set aside in favor of the final submission.

---

## Try 3

The second real attempt at the problem, tried after the MossFormer2 direction in Try 2 was set aside for taking too long to train. The question here was narrower: could SepFormer's public three speaker checkpoint be adapted to handle four speakers directly, within a one day compute budget.

### What was tried

**1. Zero training baseline**, built as a fallback and a sanity check: run the four speaker mixture through the pretrained three speaker SepFormer, take whichever of its three output streams still looks like it contains two people, and split that stream again using a pretrained two speaker SepFormer. This gave a working four speaker output with no training at all, measuring **7.6 dB** of separation improvement on a fixed test set, the number everything else had to beat.

**2. Checkpoint surgery**, the actual goal of this attempt: modify the three speaker checkpoint itself so it could output four separated streams directly. In SepFormer, the speaker count is set by a single layer that splits shared internal features into one slice per speaker, while the rest of the network only encodes general properties of speech. This layer was widened and given a fourth slice, with the three original pretrained slices copied over so most of the model stayed intact, and only the new part was fine tuned on synthetically mixed four speaker audio.

### Why it was set aside

This is where the attempt ran into the problems that eventually ended it:

- Kaggle's default GPU turned out to be an older card that current PyTorch builds no longer support, so the model would load but crash on its first real operation. This took real time to diagnose since the error gave no obvious hint that the issue was hardware, not code.
- Idle kernel restarts on Kaggle wiped loaded models and variables from memory while leaving the notebook looking untouched, producing confusing errors on code that had run correctly minutes before.
- Mixed precision training needed a specific fix to avoid silently corrupting the loss calculation.
- The model's own built in inference method could not be used after the layer swap, since it still assumed three speakers, requiring a separate custom function just to get predictions out.
- Training was limited to a single sample at a time due to memory limits, over roughly one GPU day of compute, far below what this kind of fine tuning normally needs to fully converge.

Given how much time went into these environment and infrastructure problems rather than into separation quality itself, and with the fine tuning still incomplete and not reliably better than the zero training baseline within the time available, this direction was dropped in favor of committing fully to the recursive, pretrained only cascade approach, which is what the final submission is built on.

### Possible extensions 
The most natural next step is making speaker counting fully automatic end to end, since the diarization trigger already behaves like a primitive counter. Beyond that, the cascade would benefit from a separator fine tuned on noisier, more reverberant mixtures so the recursion holds up outside clean studio style audio. The checkpoint surgery from Try 3 also remains worth finishing on a larger compute budget, since a native four or five speaker model would remove the artifact stacking that recursion introduces.
