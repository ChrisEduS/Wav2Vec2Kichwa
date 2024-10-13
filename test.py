def collate_fn(batch):
    vocab_file = 'vocab.json'
    tokenizer = Wav2Vec2CTCTokenizer(vocab_file, unk_token='[UNK]', pad_token='[PAD]', word_delimiter_token='|')
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)

    audio = [sample['audio'] for sample in batch] 
    
    transcription = [sample['transcription'] for sample in batch]

    # process audios with padding and normalization
    inputs = processor(audio, sampling_rate=16000, return_tensors='pt', padding=True)
    
    # Obtener input_values y attention_mask
    input_values = inputs.input_values
    attention_mask = inputs.attention_mask

    # Procesar las transcripciones con padding
    with processor.as_target_processor():
        labels = processor(transcription, return_tensors='pt', padding=True).input_ids

    # Mantener los datos adicionales
    eaf_paths = [sample['eaf_path'] for sample in batch]
    durations = [sample['duration'] for sample in batch]
    fs = [sample['fs'] for sample in batch]

    return {
        'input_values': input_values,       # Audios procesados
        'attention_mask': attention_mask,   # Máscara de atención
        'labels': labels,                   # Transcripciones procesadas
        'audio': audio,
        'transcription': transcription,
        'eaf_path': eaf_paths,              # Rutas adicionales
        'duration': durations,              # Duraciones de los audios
        'fs': fs                            # Frecuencia de muestreo
    }

# --- previous collate_fn 
def collate_fn(batch):
    """
    Collate function to pad audio tensors to the same length.
    """
    # Obtener la longitud máxima del audio en el lote
    max_audio_len = max([sample['audio'].size(0) for sample in batch])
    
    # Crear tensores rellenos y organizar los datos
    audio_tensors = []
    for sample in batch:
        audio = sample['audio']
        pad_size = max_audio_len - audio.size(0)
        padded_audio = torch.nn.functional.pad(audio, (0, pad_size))
        audio_tensors.append(padded_audio)
    
    # Concatenar los audios en un solo tensor
    audios = torch.stack(audio_tensors)
    
    # Organizar las demás entradas del batch
    transcriptions = [sample['transcription'] for sample in batch]
    eaf_paths = [sample['eaf_path'] for sample in batch]
    durations = [sample['duration'] for sample in batch]
    fs = [sample['fs'] for sample in batch]

    return {
        'audio': audios,
        'transcription': transcriptions,
        'eaf_path': eaf_paths,
        'duration': durations,
        'fs': fs
    }