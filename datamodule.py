import os
import lightning as L
import utils
import json
from eaf_manager import EafManager
from torch.utils.data import random_split, DataLoader

class DataModule_KichwaWav2vec2(L.LightningDataModule):

    def __init__(self, data_dir: str, processed_data_dir: str, batch_size: int):
        # Load entire dataset
        super().__init__()
        # main data dirs
        self.data_dir = data_dir
        self.pdata_dir = processed_data_dir
        
        #dirs for train and test
        self.train_dir = self.data_dir+'/train'
        self.test_dir = self.data_dir+'/test'

        self.ptrain_dir = self.pdata_dir+'/train'
        self.ptest_dir = self.pdata_dir+'/test'
        
        self.batch_size=batch_size
        self.eaf_manager = EafManager()

    def prepare_data(self):
        master_train_json = self.data_dir+'/train.json'
        master_test_json = self.data_dir='/test.json'
        # TRAINING DATA --------------------------
        # extracting metadata of master files
        print('- PROCESSING TRAINING DATA -')
        self.eaf_manager.extract_metadata(
            data_dir=self.train_dir,
            output_json_path=master_train_json,
            transcription_ext='eaf',
            audio_ext='wav'            
        )
        # segment eaf and audio and generates their metadata including transcription
        self.eaf_manager.make_data(
            input_json_file=master_train_json,
            output_dir=self.ptrain_dir,
            output_json_file=self.pdata_dir+'/train.json',
            input_audio_ext='wav',
            output_audio_ext='wav'
        )
        # TESTING DATA -------------------------
        # extracting metadata of master files
        print('- PROCESSING TESTING DATA -')
        self.eaf_manager.extract_metadata(
            data_dir=self.test_dir,
            output_json_path=master_test_json,
            transcription_ext='eaf',
            audio_ext='wav',
        )
        # segment eaf and audio and generates their metadata including transcription
        self.eaf_manager.make_data(
            input_json_file=master_test_json,
            output_json_file=self.pdata_dir+'/test.json',
            output_dir=self.ptest_dir,
            input_audio_ext='wav',
            output_audio_ext='wav'
        )

    def setup(self, stage: str):
        pass
        # # Assign train/val datasets for use in dataloaders
        # if stage == "fit":
        #     mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        #     self.mnist_train, self.mnist_val = random_split(
        #         mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        #     )


        # # Assign test dataset for use in dataloader(s)
        # if stage == "test":
        #     self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        # if stage == "predict":
        #     self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)
    
    
    

