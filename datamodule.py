import os
import torch
import json
import lightning as L
from eaf_manager import EafManager
from torch.utils.data import random_split, DataLoader
from dataset import KichwaAudioDataset  # Ensure this is the correct path

class DataModule_KichwaWav2vec2(L.LightningDataModule):
    
    def __init__(self, data_dir: str, processed_data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.pdata_dir = processed_data_dir
        self.batch_size = batch_size
        self.eaf_manager = EafManager()
        
        # Paths for train test directories
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.test_dir = os.path.join(self.data_dir, 'test')

        # Paths for processed train test directories
        self.ptrain_dir = os.path.join(self.data_dir, 'train')
        self.ptest_dir = os.path.join(self.data_dir, 'test')
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

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
        # excluding data based on their duration (ms) look up the function. 
        ptrain_json = os.path.join(self.pdata_dir, 'train.json')
        ftrain_json = os.path.join(self.pdata_dir, 'final_train.json')
        self.filter_json_by_duration(input_json_path=ptrain_json, 
                                    output_json_path=ftrain_json)

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
        # excluding data with duration outside 3000ms and 7000ms
        ptest_json = os.path.join(self.pdata_dir, 'test.json')
        ftest_json = os.path.join(self.pdata_dir, 'final_test.json')
        self.filter_json_by_duration(input_json_path=ptest_json, 
                                    output_json_path=ftest_json)

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
            full_dataset = KichwaAudioDataset(json_file=final_train_json)

            # Split into train and validation datasets (80% train, 20% validation)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            # Load test dataset
            self.test_dataset = KichwaAudioDataset(json_file=final_test_json)
        
        if stage == 'predict' or stage is None:
            # The same dataset can be used for predictions if necessary
            self.predict_dataset = KichwaAudioDataset(json_file=final_test_json)

    def train_dataloader(self):
        """Return DataLoader for the training set."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    def val_dataloader(self):
        """Return DataLoader for the validation set."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    def test_dataloader(self):
        """Return DataLoader for the test set."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    def predict_dataloader(self):
        """Return DataLoader for the predict set."""
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    def filter_json_by_duration(self, input_json_path, output_json_path, min_duration=1500, max_duration=9000):
        print('Filtrando datos por duracion (ms)')
        # Cargar el JSON desde el archivo
        with open(input_json_path, 'r') as infile:
            data = json.load(infile)

        # Filtrar los registros cuyo duration_ms esté entre min_duration y max_duration
        filtered_data = [entry for entry in data if min_duration <= entry['duration_ms'] <= max_duration]

        # Guardar el JSON filtrado en el archivo de salida
        with open(output_json_path, 'w') as outfile:
            json.dump(filtered_data, outfile, indent=4)

        print(f"Filtrado completado. Se guardó el archivo filtrado en: {output_json_path}")



def collate_fn(batch):
    """
    Collate function to pad audio tensors to the same length.
    """
    # Obtener la longitud máxima del audio en el lote
    max_audio_len = max([sample['audio'].size(1) for sample in batch])
    
    # Crear tensores rellenos y organizar los datos
    audio_tensors = []
    for sample in batch:
        audio = sample['audio']
        pad_size = max_audio_len - audio.size(1)
        padded_audio = torch.nn.functional.pad(audio, (0, pad_size))
        audio_tensors.append(padded_audio)
    
    # Concatenar los audios en un solo tensor
    audios = torch.stack(audio_tensors)
    
    # Organizar las demás entradas del batch
    transcriptions = [sample['transcription'] for sample in batch]
    eaf_paths = [sample['eaf_path'] for sample in batch]
    durations = [sample['duration'] for sample in batch]
    fs = [sample['fs'] for sample in batch]

    return {
        'audio': audios,
        'transcription': transcriptions,
        'eaf_path': eaf_paths,
        'duration': durations,
        'fs': fs
    }
