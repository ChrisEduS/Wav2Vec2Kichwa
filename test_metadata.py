import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer
from torch.utils.data import DataLoader

class KichwaAudioDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Wav2Vec2FineTuning(pl.LightningModule):
    def __init__(self, lr: float, dataset, processor_name: str, batch_size: int):
        super(Wav2Vec2FineTuning, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.processor = Wav2Vec2Processor.from_pretrained(processor_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(processor_name)
        self.dataset = dataset

    def forward(self, inputs, input_lengths):
        outputs = self.model(input_values=inputs, attention_mask=input_lengths).logits
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        audio = batch['audio']
        transcriptions = batch['transcription']

        # Preprocess audio and transcriptions
        inputs = self.processor(audio, sampling_rate=batch['fs'][0], return_tensors="pt", padding=True)
        input_values = inputs.input_values.squeeze(1).to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Encode transcriptions to target labels
        with self.processor.as_target_processor():
            targets = self.processor(transcriptions, return_tensors="pt", padding=True).input_ids

        # Forward pass through the model
        logits = self.forward(input_values, attention_mask)

        # Compute loss (CTC loss)
        loss = F.ctc_loss(
            logits.transpose(0, 1),  # (T, N, C) format required for CTC
            targets,
            input_lengths=attention_mask.sum(-1),  # Mask for variable-length inputs
            target_lengths=torch.tensor([len(t) for t in targets], device=self.device),
            blank=self.processor.tokenizer.pad_token_id,
            zero_infinity=True,
        )

        self.log("train_loss", loss, on_epoch=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        audio = batch['audio']
        transcriptions = batch['transcription']

        # Preprocess audio
        inputs = self.processor(audio, sampling_rate=batch['fs'][0], return_tensors="pt", padding=True)
        input_values = inputs.input_values.squeeze(1).to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Forward pass
        logits = self.forward(input_values, attention_mask)
        pred_ids = torch.argmax(logits, dim=-1)

        # Decode predictions and compute WER
        pred_transcriptions = self.processor.batch_decode(pred_ids)
        avg_wer = wer(transcriptions, pred_transcriptions)

        self.log("val_wer", avg_wer, on_epoch=True, batch_size=self.batch_size)
        return {"val_wer": avg_wer}

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)


# Example of how to use the KichwaAudioDataset and Wav2Vec2FineTuning class
if __name__ == "__main__":
    # Example dataset (replace with actual loading)
    dataset = KichwaAudioDataset([
        {
            'audio': torch.randn(1, 16000),  # Example 1-second audio tensor
            'transcription': "example transcription",
            'duration': 1000,
            'fs': 16000,
            'eaf_path': 'path/to/eaf',
        },
        # Add more data samples
    ])

    # Define the model
    model = Wav2Vec2FineTuning(
        lr=3e-5,
        dataset=dataset,
        processor_name="facebook/wav2vec2-large-960h",
        batch_size=4
    )

    # Define a trainer
    trainer = pl.Trainer(max_epochs=3)

    # Train the model
    trainer.fit(model)
