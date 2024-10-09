from jiwer import cer  # Import Character Error Rate function

class Wav2Vec2FineTuner(L.LightningModule):
    def __init__(self, model_name, vocab_file, batch_size, learning_rate=1e-4):
        super().__init__()
        self.lr = learning_rate
        self.batch_size = batch_size
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.tokenizer = Wav2Vec2CTCTokenizer(vocab_file, unk_token='[UNK]', pad_token='[PAD]', word_delimiter_token='|')
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        self.processor = Wav2Vec2Processor(self.feature_extractor, self.tokenizer)

    def forward(self, inputs, attention_mask):
        outputs = self.model(input_values=inputs, attention_mask=attention_mask).logits
        return outputs

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
        loss = torch.nn.functional.ctc_loss(
            logits.transpose(0, 1),  # (T, N, C) format required for CTC
            targets,
            input_lengths=attention_mask.sum(-1),  # Mask for variable-length inputs
            target_lengths=torch.tensor([len(t) for t in targets], device=self.device),
            blank=self.processor.tokenizer.pad_token_id,
            zero_infinity=True,
        )

        # Decode predictions
        pred_ids = torch.argmax(logits, dim=-1)
        pred_transcriptions = self.processor.batch_decode(pred_ids)

        # Compute WER and CER
        avg_wer = wer(transcriptions, pred_transcriptions)
        avg_cer = cer(transcriptions, pred_transcriptions)

        # Log loss, WER, and CER
        self.log("train_loss", loss, on_epoch=True, batch_size=self.batch_size)
        self.log("train_wer", avg_wer, on_epoch=True, batch_size=self.batch_size)
        self.log("train_cer", avg_cer, on_epoch=True, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        audio = batch['audio']
        transcriptions = batch['transcription']

        # Preprocess audio
        inputs = self.processor(audio, sampling_rate=batch['fs'][0], return_tensors="pt", padding=True)
        input_values = inputs.input_values.squeeze(1).to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Encode transcriptions to target labels
        with self.processor.as_target_processor():
            targets = self.processor(transcriptions, return_tensors="pt", padding=True).input_ids

        # Forward pass
        logits = self.forward(input_values, attention_mask)
        pred_ids = torch.argmax(logits, dim=-1)

        # Compute loss (CTC loss)
        val_loss = torch.nn.functional.ctc_loss(
            logits.transpose(0, 1),  # (T, N, C) format required for CTC
            targets,
            input_lengths=attention_mask.sum(-1),  # Mask for variable-length inputs
            target_lengths=torch.tensor([len(t) for t in targets], device=self.device),
            blank=self.processor.tokenizer.pad_token_id,
            zero_infinity=True,
        )

        # Decode predictions
        pred_transcriptions = self.processor.batch_decode(pred_ids)

        # Compute WER and CER
        avg_wer = wer(transcriptions, pred_transcriptions)
        avg_cer = cer(transcriptions, pred_transcriptions)

        # Log validation loss, WER, and CER
        self.log("val_loss", val_loss, on_epoch=True, batch_size=self.batch_size)
        self.log("val_wer", avg_wer, on_epoch=True, batch_size=self.batch_size)
        self.log("val_cer", avg_cer, on_epoch=True, batch_size=self.batch_size)

        return {"val_loss": val_loss, "val_wer": avg_wer, "val_cer": avg_cer}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
