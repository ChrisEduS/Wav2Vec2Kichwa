    def _shared_step(self, batch):
        
        transcriptions = batch['transcription']
        input_values = batch['input_values']
        attention_mask = batch['attention_mask']
        targets = batch['labels']
        
        print(input_values.shape)
        print(attention_mask.shape)
        print(targets.shape)
        print(self.model.config.inputs_to_logits_ratio)

        # Preprocess audio
        # inputs = self.processor(audio, sampling_rate=self.fs, return_tensors="pt", padding=True, return_attention_mask=True)
        # input_values = inputs.input_values[0].to(self.device)
        # attention_mask = inputs.attention_mask.to(self.device)
        
        # Encode transcriptions to target labels
        # with self.processor.as_target_processor():
        #     targets = self.processor(transcriptions, return_tensors="pt", padding=True).input_ids
        
        # Forward pass
        logits = self.forward(input_values, attention_mask)
        print('logits transposed shape: ', logits.transpose(0,1).shape)

        print('input_lengths: ', attention_mask.sum(-1))
        print('traget_lengths: ', torch.tensor([len(t) for t in targets]))

        #Compute CTC loss
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
        pred_transcriptions = self.processor.batch_decode(pred_ids, group_tokens=False)

        # Compute WER and CER
        avg_wer = wer(transcriptions, pred_transcriptions)
        avg_cer = cer(transcriptions, pred_transcriptions)

        return loss, avg_wer, avg_cer
