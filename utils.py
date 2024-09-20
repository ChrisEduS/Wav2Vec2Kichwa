params = {
            'data_dir': '/root/Wav2Vec2Kichwa/data',
            'processed_data_dir': '/root/Wav2Vec2Kichwa/segmented_data',
            'train_audio_ext': '.wav',
            'test_audio_ext': '.mp4', 
            'train_trans_ext': '.eaf',
            'test_trans_ext': '.eaf',
            'batch_size': 32
        }
master_files = {
            'audio': [f'{i:02} MASTER K{i}.mp4' for i in range(1, 21)],
            'elan': [f'Chapter{i}.eaf' for i in range(1, 21)]
}