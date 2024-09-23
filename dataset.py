import json
import torch
import torchaudio
from torch.utils.data import Dataset

class KichwaAudioDataset(Dataset):
    def __init__(self, json_file):
        """
        Args:
            json_file (str): Path to the JSON file containing the dataset.
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing audio tensor, transcription, and metadata.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data[idx]
        
        audio_path = sample['audio_path']
        transcription = sample['transcription']
        duration = sample['duration_ms']
        fs = sample['fs']
        eaf_path = sample['eaf_path']


        # Load the audio file using torchaudio
        audio, sample_rate = torchaudio.load(audio_path)

        # If the sample rate in the JSON does not match the actual sample rate, resample
        if sample_rate != fs:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=fs)
            audio = resampler(audio)
        
        return {
            'audio': audio,                 # Audio tensor loaded
            'transcription': transcription, # transcription text
            'duration': duration,           # Duration in milliseconds
            'fs': fs,                       # Sample rate (after resampling if needed)
            'eaf_path': eaf_path            # path for eaf file. Maybe not used. 
        }
