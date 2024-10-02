import pytorch_lightning as L
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from torch.optim import AdamW
from torch.nn.functional import cross_entropy

class Wav2Vec2FineTuner(L.LightningModule):
    def __init__(self, model_name, vocab_file, learning_rate=1e-4):
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.tokenizer = Wav2Vec2CTCTokenizer(vocab_file, unk_token='[UNK]', pad_token='[PAD]', word_delimiter_token='|')
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        self.processor = Wav2Vec2Processor(self.feature_extractor, self.tokenizer)
    def forward(self):
        pass
    def training_step(self):
        pass
    def validation_step(self):
        pass
    def configure_optimizers(self):
        pass
