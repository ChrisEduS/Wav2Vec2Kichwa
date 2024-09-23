import os
from torch.utils.data import random_split, DataLoader
from dataset import KichwaAudioDataset  # Ensure this is the correct path

class DataModule_KichwaWav2vec2(L.LightningDataModule):
    
    def __init__(self, data_dir: str, processed_data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.pdata_dir = processed_data_dir
        self.batch_size = batch_size
        
        # Paths for train and test data
        self.train_json = os.path.join(self.pdata_dir, 'train.json')
        self.test_json = os.path.join(self.pdata_dir, 'test.json')
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def setup(self, stage: str):
        """
        Load datasets and perform any necessary preprocessing steps.
        The `stage` can be either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage == 'fit' or stage is None:
            # Load full training dataset from the preprocessed data JSON file
            full_dataset = KichwaAudioDataset(json_file=self.train_json)

            # Split into train and validation datasets (80% train, 20% validation)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            # Load test dataset
            self.test_dataset = KichwaAudioDataset(json_file=self.test_json)
        
        if stage == 'predict' or stage is None:
            # The same dataset can be used for predictions if necessary
            self.predict_dataset = KichwaAudioDataset(json_file=self.test_json)

    def train_dataloader(self):
        """Return DataLoader for the training set."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """Return DataLoader for the validation set."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        """Return DataLoader for the test set."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def predict_dataloader(self):
        """Return DataLoader for the predict set."""
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
