import utils
import torch
import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
from datamodule import DataModule_KichwaWav2vec2 
from torchmodule import Wav2Vec2FineTuner

def main():
    # Configuraciones básicas
    dirs = utils.dirs
    configs = utils.configs

    data_dir = dirs['data_dir']
    processed_data_dir = dirs['processed_data_dir']
    checkpoints_dir = dirs['checkpoints']

    freq_sample = configs['fs']
    batch_size = configs['batch_size'] 
    lr = configs['lr']

    # Instantiate and prepare data module
    datamodule = DataModule_KichwaWav2vec2(data_dir=data_dir,
                                            processed_data_dir=processed_data_dir,
                                            freq_sample=freq_sample,
                                            batch_size=batch_size)

    #     # Preparar los datos (solo si no has procesado los datos antes)
    # datamodule.prepare_data()
    
    # # Configurar el DataModule para el entrenamiento
    # datamodule.setup(stage='fit')

    # # Obtener el train_dataloader
    # test_loader = datamodule.train_dataloader()
    # dataset = test_loader.dataset

    # print(dataset[0]['audio'], dataset[0]['audio'].shape) 
    # print('Cantidad de datos para testear:', len(test_loader.dataset))
# --------------------------------------------------------------------------------------------------
    # Instantiate model
    model_name = 'facebook/wav2vec2-xls-r-300m'
    model = Wav2Vec2FineTuner(model_name=model_name,
                              vocab_file='vocab.json',
                              fs=freq_sample,
                              batch_size=batch_size,
                              learning_rate=lr)
    
    # init wandb 
    wandb.init(project='Wav2Vec2KichwaFineTuner')

    # callbacks
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=checkpoints_dir,
                                                              filename='{epoch}-{val_loss:.2f}-{val_wer:.2f}-{val_cer:.2f}',
                                                              save_top_k=5,
                                                              save_last=True, monitor="val_loss", mode='min')
    early_stopping_callback = L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")

    
    # model trainer
    trainer = L.Trainer(
        max_epochs=20,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        strategy='auto',
        precision=16,
        accumulate_grad_batches=2,
        gradient_clip_val=0.5,
        sync_batchnorm=True,
        profiler="simple",  # Optional for debugging
        logger=WandbLogger(project='Wav2Vec2KichwaFineTuner')
    )

    trainer.fit(model=model,
                datamodule=datamodule)
    
    # EXAMPLE
    # Set model to evaluation mode
    # model.eval()

    # # Get a sample from the test dataset
    # sample = datamodule.test_dataloader().dataset[0]  # First sample from test dataset
    # audio = sample['audio']
    # transcription = sample['transcription']

    # # Print the ground truth transcription
    # print(f"Ground Truth: {transcription}")

    # # Preprocess the audio input for the model
    # inputs = model.processor(audio, sampling_rate=sample['fs'], return_tensors="pt", padding=True)
    # input_values = inputs.input_values.squeeze(1).to(model.device)
    # attention_mask = inputs.attention_mask.to(model.device)

    # # Forward pass through the model
    # with torch.no_grad():
    #     logits = model(input_values, attention_mask=attention_mask)

    # # Decode the predicted token IDs into text
    # pred_ids = torch.argmax(logits, dim=-1)
    # predicted_transcription = model.processor.batch_decode(pred_ids)

    # # Print predicted transcription
    # print(f"Predicted Transcription: {predicted_transcription[0]}")


if __name__ == "__main__":
    main()
