import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
import config

class PostProcessor:
    def __init__(self):
        print("Loading ECAPA-TDNN model for speaker deduplication...")
        self.encoder = EncoderClassifier.from_hparams(
            source=config.ECAPA_MODEL,
            savedir=config.ECAPA_SAVEDIR
        )

    def _compute_rms_db(self, waveform):
        """
        Computes RMS energy in dBFS.
        waveform: shape (1, time)
        """
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms == 0:
            return -float('inf')
        return 20 * torch.log10(rms).item()

    def prune_silence(self, channels_dict):
        """
        Drops channels with RMS energy below SILENCE_RMS_THRESHOLD_DB.
        channels_dict: { 'id': waveform_tensor }
        Returns: surviving_channels_dict, dropped_ids
        """
        surviving = {}
        dropped = []
        for cid, wav in channels_dict.items():
            rms_db = self._compute_rms_db(wav)
            if rms_db >= config.SILENCE_RMS_THRESHOLD_DB:
                surviving[cid] = wav
            else:
                dropped.append((cid, rms_db))
        return surviving, dropped

    def extract_embedding(self, waveform):
        """
        Extracts speaker embedding using ECAPA-TDNN.
        ECAPA expects 16kHz audio, we must resample if it's 8kHz.
        waveform: shape (1, time) at SEPFORMER_SAMPLE_RATE (8kHz)
        """
        # Resample 8kHz to 16kHz for ECAPA
        if config.SEPFORMER_SAMPLE_RATE != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=config.SEPFORMER_SAMPLE_RATE,
                new_freq=16000
            )
            wav_16k = resampler(waveform)
        else:
            wav_16k = waveform
            
        with torch.no_grad():
            emb = self.encoder.encode_batch(wav_16k)
        # emb shape is (batch, 1, channels) -> (1, 1, 192)
        return emb.squeeze()

    def deduplicate(self, channels_dict):
        """
        Greedy pairwise merging of channels with high cosine similarity.
        channels_dict: { 'id': waveform_tensor }
        Returns: deduplicated_dict, merges
        """
        if not channels_dict:
            return {}, []

        # 1. Compute embeddings and RMS for all channels
        embeddings = {}
        energies = {}
        for cid, wav in channels_dict.items():
            embeddings[cid] = self.extract_embedding(wav)
            energies[cid] = self._compute_rms_db(wav)
            
        # 2. Greedy merge
        cids = list(channels_dict.keys())
        merged_away = set()
        merges = []
        
        # Sort by energy descending, so we prefer keeping the louder signal
        cids.sort(key=lambda c: energies[c], reverse=True)
        
        from torch.nn.functional import cosine_similarity
        
        for i in range(len(cids)):
            c1 = cids[i]
            if c1 in merged_away:
                continue
            for j in range(i + 1, len(cids)):
                c2 = cids[j]
                if c2 in merged_away:
                    continue
                
                sim = cosine_similarity(embeddings[c1].unsqueeze(0), embeddings[c2].unsqueeze(0)).item()
                if sim >= config.COSINE_SIMILARITY_THRESHOLD:
                    # c1 has higher energy (due to sort), so we keep c1 and drop c2
                    merged_away.add(c2)
                    merges.append((c1, c2, sim))
                    
        dedup_dict = {cid: channels_dict[cid] for cid in cids if cid not in merged_away}
        return dedup_dict, merges

    def normalize_loudness(self, waveform):
        """Peak normalization to TARGET_NORMALIZATION_PEAK"""
        peak = torch.max(torch.abs(waveform))
        if peak > 0:
            return waveform * (config.TARGET_NORMALIZATION_PEAK / peak)
        return waveform
