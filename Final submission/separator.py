import torch
from speechbrain.inference.separation import SepformerSeparation
import config

class Separator:
    def __init__(self):
        print("Loading SepFormer model...")
        self.model = SepformerSeparation.from_hparams(
            source=config.SEPFORMER_MODEL,
            savedir=config.SEPFORMER_SAVEDIR
        )

    def separate_tensor(self, mix_tensor):
        """
        Separates an input audio tensor into 3 channels.
        mix_tensor: shape (1, time)
        Returns: tuple of 3 tensors, each of shape (1, time)
        """
        # separate_batch expects shape (batch, time)
        # Returns shape (batch, time, num_sources)
        with torch.no_grad():
            est_sources = self.model.separate_batch(mix_tensor)
        
        # est_sources has shape (1, time, 3) since wsj03mix is a 3-speaker model
        # We need to split it into 3 separate tensors
        sources = [est_sources[:, :, i] for i in range(est_sources.shape[2])]
        return sources
