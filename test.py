import os
import torch
import utils
import lightning as L
from torchmodule import Wav2Vec2FineTuner
from datamodule import DataModule_KichwaWav2vec2

# Configuración de directorios y parámetros
dirs = utils.dirs
configs = utils.configs

data_dir = dirs['data_dir']
processed_data_dir = dirs['processed_data_dir']
vocab = dirs['vocab']
checkpoints_dir = dirs['checkpoints']

freq_sample = configs['fs']
batch_size = configs['batch_size']
lr = configs['lr']

# Definir dispositivo (usa 'cuda' si tienes una GPU disponible)
# device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# Best model path
best_model = '/root/Wav2Vec2Kichwa/best_models/no-data-augmentation/electric-fire-179/epoch=42-val_loss=0.014-val_wer=0.029-val_cer=0.004-val_mer=0.029.ckpt'
model_name = 'facebook/wav2vec2-xls-r-300m'
# Cargar el modelo desde el checkpoint
model = Wav2Vec2FineTuner.load_from_checkpoint(checkpoint_path=best_model,
                                               model_name=model_name,
                                               vocab_file=vocab,
                                               fs=freq_sample,
                                               batch_size=batch_size,
                                               learning_rate=lr)

# Mover el modelo al dispositivo
# model = model.to(device)

# Instanciar y preparar el DataModule
datamodule = DataModule_KichwaWav2vec2(data_dir=data_dir,
                                       processed_data_dir=processed_data_dir,
                                       vocab=vocab,
                                       freq_sample=freq_sample,
                                       batch_size=batch_size)

trainer = L.Trainer(accelerator="gpu", devices=1)
trainer.test(model=model, datamodule=datamodule)
