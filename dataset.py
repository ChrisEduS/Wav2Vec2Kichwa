import json
import torch
import torchaudio
from torch.utils.data import Dataset

class KichwaAudioDataset(Dataset):
    def __init__(self, json_file: str, freq_sample: int):
        """
        Args:
            json_file (str): Path to the JSON file containing the dataset.
        """
        self.model_sample_rate = freq_sample
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

        # Resampling to the fs used for pretrained model. wav2vec2 this case
        
        audio = torchaudio.transforms.Resample(sample_rate, self.model_sample_rate)(audio)
        # downmix stereo to mono if needed 
        if audio.shape[0] == 2:
            audio = torch.mean(audio, dim=0)
        # deleting singleton dimension
        audio = audio.squeeze(0)
        
        
        return {
            'audio': audio,                 # Audio tensor loaded
            'transcription': transcription, # transcription text
            'duration': duration,           # Duration in milliseconds (int)
            'fs': self.model_sample_rate,   # Sample rate. Wav2Vec2 model trained with 16000 frequency rate
            'eaf_path': eaf_path            # path for eaf file. Maybe not used. 
        }

def downmix_to_mono(audio):
    # Assuming audio has shape [batch_size, 2, audio_length] (stereo)
    if audio.shape[1] == 2:
        audio = torch.mean(audio, dim=1)  # Downmix to mono by averaging across the channel dimension
    return audio