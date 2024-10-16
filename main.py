import utils
import torch
import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
from datamodule import DataModule_KichwaWav2vec2 
from torchmodule import Wav2Vec2FineTuner

def main():
    # Directories and configs
    dirs = utils.dirs
    configs = utils.configs

    data_dir = dirs['data_dir']
    processed_data_dir = dirs['processed_data_dir']
    vocab = dirs['vocab']
    checkpoints_dir = dirs['checkpoints']

    freq_sample = configs['fs']
    batch_size = configs['batch_size'] 
    lr = configs['lr']

    # Instantiate and prepare data module
    datamodule = DataModule_KichwaWav2vec2(data_dir=data_dir,
                                            processed_data_dir=processed_data_dir,
                                            vocab = vocab,
                                            freq_sample=freq_sample,
                                            batch_size=batch_size)

# --------------------------------------------------------------------------------------------------
    # Instantiate model
    model_name = 'facebook/wav2vec2-xls-r-300m'
    model = Wav2Vec2FineTuner(model_name=model_name,
                              vocab_file=vocab,
                              fs=freq_sample,
                              batch_size=batch_size,
                              learning_rate=lr)
    
    # init wandb. resume and id flags only for resume training
    wandb.init(project='Wav2Vec2KichwaFineTuner', resume='must', id='mf1nwc3h')

    # callbacks
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=checkpoints_dir,
                                                              filename='{epoch}-{val_loss:.2f}-{val_wer:.2f}-{val_cer:.2f}',
                                                              save_top_k=5,
                                                              save_last=True, monitor="val_loss", mode='min')
    early_stopping_callback = L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")

    
    # model trainer
    trainer = L.Trainer(
        max_epochs=50,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[2, 3],
        strategy='ddp_find_unused_parameters_true',
        sync_batchnorm=True,
        profiler="simple",  # Optional for debugging
        logger=WandbLogger(project='Wav2Vec2KichwaFineTuner')
    )
    # fit model
    trainer.fit(model=model,
                datamodule=datamodule,
                ckpt_path='./checkpoints/last.ckpt') # just if you want to resume training


if __name__ == "__main__":
    main()
