import os
import torch
import utils
import lightning as L
from torchmodule import Wav2Vec2FineTuner
from datamodule import DataModule_KichwaWav2vec2
import numpy as np
import wandb

def upload_predictions_to_wandb(model, test_loader, device, project_name, sample_rate=16000, num_samples=4):
    # Initialize W&B run
    run = wandb.init(project=project_name, job_type="evaluation")
    
    # Create a W&B table with columns for audio, predicted transcription, and ground truth transcription
    columns = ["audio", "predicted_transcription", "ground_truth"]
    table = wandb.Table(columns=columns)

    # Ensure the model is in eval mode and disable gradient calculations
    model.eval()
    all_predictions = []
    all_transcriptions = []
    all_audio_paths = []

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to the device
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            transcription = batch['transcription']
            audio_path = batch['audio_path']

            # Generate predictions
            outputs = model(inputs=input_values, attention_mask=attention_mask, targets=labels).logits
            pred_ids = torch.argmax(outputs, dim=-1)
            predicted_transcriptions = model.processor.batch_decode(pred_ids)

            # Add predictions and ground truth to respective lists
            all_predictions.extend(predicted_transcriptions)
            all_transcriptions.extend(transcription)
            all_audio_paths.extend(audio_path)

    # Pick random samples to add to the W&B table
    random_indices = np.random.choice(len(all_predictions), num_samples, replace=False)
    
    for idx in random_indices:
        audio_file = all_audio_paths[idx]
        predicted_transcription = all_predictions[idx]
        ground_truth_transcription = all_transcriptions[idx]

        # Add audio, predicted transcription, and ground truth to the W&B table
        table.add_data(
            wandb.Audio(audio_file, sample_rate=sample_rate, caption=f"Sample {idx + 1}"), 
            predicted_transcription, 
            ground_truth_transcription
        )

    # Log the table to W&B
    run.log({"Test-table": table})

    # Finish the W&B run
    run.finish()

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
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
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

# Project name
project_name = "Wav2Vec2KichwaFineTuner"

# Example usage
upload_predictions_to_wandb(model, test_loader, device, project_name)
