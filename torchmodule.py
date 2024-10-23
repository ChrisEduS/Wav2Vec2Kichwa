import lightning as L
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from jiwer import wer, cer, mer

class Wav2Vec2FineTuner(L.LightningModule):
    def __init__(self, model_name: str, vocab_file: str, fs: int, batch_size: int, learning_rate: int):
        super().__init__()
        self.fs = fs
        self.lr = learning_rate

        self.tokenizer = Wav2Vec2CTCTokenizer(vocab_file, unk_token='[UNK]', pad_token='[PAD]', word_delimiter_token='|')
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        self.processor = Wav2Vec2Processor(self.feature_extractor, self.tokenizer)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name,
                                     vocab_size=len(self.processor.tokenizer),
                                     ctc_loss_reduction="mean",
                                     apply_spec_augment=False, # No data augmentation
                                     pad_token_id=self.processor.tokenizer.pad_token_id,
                                     bos_token_id=self.processor.tokenizer.bos_token_id,
                                     eos_token_id=self.processor.tokenizer.eos_token_id
                                     )
        self.model.freeze_feature_extractor()

        # check model configs
        # print(self.model.config)

        self.log_args = {
            'on_step': True,
            'on_epoch': True,
            'logger': True,
            'prog_bar': True
        }
    
    def forward(self, inputs, attention_mask, targets):
        outputs = self.model(input_values=inputs, attention_mask=attention_mask, labels=targets)
        return outputs

    def _shared_step(self, batch):
        transcriptions = batch['transcription']
        input_values = batch['input_values']
        attention_mask = batch['attention_mask']
        targets = batch['labels']

        # Forward pass
        outputs = self.forward(input_values, attention_mask, targets)
        logits = outputs.logits
        loss = outputs.loss

        # Decode predictions and compute metrics
        pred_ids = torch.argmax(logits, dim=-1)
        pred_transcriptions = self.processor.batch_decode(pred_ids)

        # Compute WER, CER, and MER
        avg_wer = wer(transcriptions, pred_transcriptions)
        avg_cer = cer(transcriptions, pred_transcriptions)
        avg_mer = mer(transcriptions, pred_transcriptions)

        return loss, avg_wer, avg_cer, avg_mer

    
    def training_step(self, batch, batch_idx):
        loss, avg_wer, avg_cer, avg_mer = self._shared_step(batch)
        self.log("train_loss", loss, **self.log_args)
        self.log("train_wer", avg_wer, **self.log_args)
        self.log("train_cer", avg_cer, **self.log_args)
        self.log("train_mer", avg_mer, **self.log_args)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, avg_wer, avg_cer, avg_mer = self._shared_step(batch)
        self.log("val_loss", loss, **self.log_args)
        self.log("val_wer", avg_wer, **self.log_args)
        self.log("val_cer", avg_cer, **self.log_args)
        self.log("val_mer", avg_mer, **self.log_args)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer