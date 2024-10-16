import os
import torch
import utils
import lightning as L
from torchmodule import Wav2Vec2FineTuner
from datamodule import DataModule_KichwaWav2vec2
import numpy as np

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Best model path
best_model = 'epoch=30-val_loss=0.02-val_wer=0.07-val_cer=0.01.ckpt'
# Cargar el modelo desde el checkpoint
best_model_path = os.path.join(checkpoints_dir, best_model)  # Cambia esto si usas otro nombre
model_name = 'facebook/wav2vec2-xls-r-300m'
model = Wav2Vec2FineTuner.load_from_checkpoint(checkpoint_path=best_model_path,
                                               model_name=model_name,
                                               vocab_file=vocab,
                                               fs=freq_sample,
                                               batch_size=batch_size,
                                               learning_rate=lr)

# Mover el modelo al dispositivo
model = model.to(device)

# Instanciar y preparar el DataModule
datamodule = DataModule_KichwaWav2vec2(data_dir=data_dir,
                                       processed_data_dir=processed_data_dir,
                                       vocab=vocab,
                                       freq_sample=freq_sample,
                                       batch_size=batch_size)

# Preparar los datos de prueba
datamodule.prepare_data()
datamodule.setup(stage='test')

# Obtener el DataLoader de prueba
test_loader = datamodule.test_dataloader()

# Poner el modelo en modo de evaluación
model.eval()

# Desactivar el cálculo de gradientes
all_predictions = []
all_transcriptions = []
all_eaf_paths = []
with torch.no_grad():
    for batch in test_loader:
        # Mover los inputs al mismo dispositivo que el modelo
        input_values = batch['input_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        transcription = batch['transcription']
        eaf_path = batch['eaf_path']

        # Hacer predicciones con el modelo
        outputs = model(inputs=input_values, attention_mask=attention_mask, targets=labels).logits
        # append outputs and transcriptions and eaf paths
        all_predictions.append(outputs)
        all_transcriptions.append(transcription)
        all_eaf_paths.append(eaf_path)

# random batch
random_idx = np.random.randint(len(all_predictions))
# get the predicted transcription
predictions_batch = all_predictions[random_idx]
pred_ids = torch.argmax(predictions_batch, dim=-1)
predicted_transcription = model.processor.batch_decode(pred_ids)
# get the true transcription
transcription_batch = all_transcriptions[random_idx]
# get the eaf path
eaf_path = all_eaf_paths[random_idx]

print(predicted_transcription)
print(transcription_batch)
print(eaf_path)
