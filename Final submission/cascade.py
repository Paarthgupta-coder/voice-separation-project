import config

class CascadePipeline:
    def __init__(self, separator, diarizer):
        self.separator = separator
        self.diarizer = diarizer

    def process(self, initial_mix):
        """
        Recursively separate the mix.
        initial_mix: shape (1, time)
        Returns: dict of { path_string: waveform_tensor }
        """
        print("\n--- Starting Recursive Separation ---")
        
        # The dictionary to hold all output channels.
        # Key: provenance string (e.g., "root_1", "root_2_1")
        # Value: waveform tensor
        results = {}
        
        # We'll use a queue for BFS processing: (waveform, path_prefix, current_depth)
        queue = [(initial_mix, "ch", 0)]
        
        while queue:
            wav, path, depth = queue.pop(0)
            
            # 1. Base separation
            print(f"Separating {path if path != 'ch' else 'root mix'} (depth {depth})...")
            sub_channels = self.separator.separate_tensor(wav)
            
            # 2. Check each sub-channel
            for i, sub_wav in enumerate(sub_channels):
                sub_path = f"{path}_{i}"
                
                # If we hit max depth, we don't check for further overlap, just keep it.
                if depth >= config.MAX_RECURSION_DEPTH:
                    results[sub_path] = sub_wav
                    continue
                    
                # Check for overlap
                speakers_est = self.diarizer.estimate_speaker_count(sub_wav)
                
                if speakers_est > 1:
                    print(f"  -> {sub_path} flagged! Detected {speakers_est} speakers. Queuing for re-split.")
                    queue.append((sub_wav, sub_path, depth + 1))
                else:
                    print(f"  -> {sub_path} clean (est. {speakers_est} speaker). Keeping.")
                    results[sub_path] = sub_wav
                    
        return results
