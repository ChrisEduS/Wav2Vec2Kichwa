import os
import utils
import lightning as L
from torch.utils.data import random_split, DataLoader
from dataset import KichwaAudioDataset 
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

class DataModule_KichwaWav2vec2(L.LightningDataModule):
    
    def __init__(self, data_dir: str, processed_data_dir: str, vocab: str, freq_sample: int, batch_size: int):
        super().__init__()
        # data directories
        self.data_dir = data_dir
        self.pdata_dir = processed_data_dir
        self.vocab = vocab
        # configs
        self.freq_sample = freq_sample
        self.batch_size = batch_size

        # datasets 
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None


    def setup(self, stage: str):
        """
        Load datasets and perform any necessary preprocessing steps.
        The `stage` can be either 'fit', 'validate', 'test', or 'predict'.
        """
        # Paths for segmented train and test data
        final_train_json = os.path.join(self.pdata_dir, 'final_train.json')
        final_test_json = os.path.join(self.pdata_dir, 'final_test.json')

        if stage == 'fit' or stage is None:
            # Load full training dataset from the preprocessed data JSON file
            full_dataset = KichwaAudioDataset(json_file=final_train_json, vocab=self.vocab, freq_sample=self.freq_sample)

            # Split into train and validation datasets (80% train, 20% validation)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            # Load test dataset
            self.test_dataset = KichwaAudioDataset(json_file=final_test_json, vocab=self.vocab, freq_sample=self.freq_sample)
        
        if stage == 'predict' or stage is None:
            # The same dataset can be used for predictions if necessary
            self.predict_dataset = KichwaAudioDataset(json_file=final_test_json, vocab=self.vocab, freq_sample=self.freq_sample)

    def train_dataloader(self):
        """Return DataLoader for the training set."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    def val_dataloader(self):
        """Return DataLoader for the validation set."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    def test_dataloader(self):
        """Return DataLoader for the test set."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    def predict_dataloader(self):
        """Return DataLoader for the predict set."""
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

def collate_fn(batch):
    dirs = utils.dirs
    vocab_file = dirs['vocab']
    tokenizer = Wav2Vec2CTCTokenizer(vocab_file, unk_token='[UNK]', pad_token='[PAD]', word_delimiter_token='|')
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)

    # Extract audio data
    audio = [sample['audio'].squeeze(0).numpy() for sample in batch]  # Convert from tensors to numpy arrays
    
    # Extract transcription data
    transcription = [sample['transcription'] for sample in batch]

    # Ensure processor correctly pads audios with padding=True
    inputs = processor(audio, sampling_rate=16000, return_tensors='pt', padding=True)
    
    # Obtain input_values and attention_mask
    input_values = inputs.input_values
    attention_mask = inputs.attention_mask

    # Process transcriptions with padding
    with processor.as_target_processor():
        labels = processor(transcription, return_tensors='pt', padding=True).input_ids

    # Replace pad token ids with -100 to ignore padding when calculating loss
    labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)

    # Keep additional data
    audio_paths = [sample['audio_path'] for sample in batch]
    eaf_paths = [sample['eaf_path'] for sample in batch]
    durations = [sample['duration'] for sample in batch]
    fs = [sample['fs'] for sample in batch]

    return {
        'input_values': input_values,       # Processed audios
        'attention_mask': attention_mask,   # Attention mask
        'labels': labels,                   # Processed transcriptions. mapped to vocab
        'audio': audio,                     # Original audios
        'transcription': transcription,     # Original transcriptions
        'duration': durations,              # Audio durations
        'fs': fs,                           # Sampling rates
        'audio_path': audio_paths,          # Audio paths
        'eaf_path': eaf_paths,              # eaf file paths
    }