import os
# import utils from the parent directory pls
import sys

# setting path
sys.path.append('../Wav2Vec2Kichwa')
import utils

import json
import lightning as L
from eaf_manager import EafManager
    

def prepare_data(data_dir, pdata_dir, vocab):
    # Paths for train test directories
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # Paths for processed train test directories
    ptrain_dir = os.path.join(pdata_dir, 'train')
    ptest_dir = os.path.join(pdata_dir, 'test')

    # EAF manager
    eaf_manager = EafManager()

    # Paths for master train test json files
    master_train_json = os.path.join(data_dir, 'train.json')
    master_test_json = os.path.join(data_dir, 'test.json')
    # TRAINING DATA --------------------------
    # extracting metadata of master files
    print('- PROCESSING TRAINING DATA -')
    eaf_manager.extract_metadata(
        data_dir=train_dir,
        output_json_path=master_train_json,
        transcription_ext='eaf',
        audio_ext='wav'            
    )
    # segment eaf and audio and generates their metadata including transcription
    eaf_manager.make_data(
        input_json_file=master_train_json,
        output_dir=ptrain_dir,
        output_json_file=pdata_dir+'/train.json',
        input_audio_ext='wav',
        output_audio_ext='wav'
    )
    # FILTERING
    # excluding data based on their duration (ms) look up the function. 
    ptrain_json = os.path.join(pdata_dir, 'train.json')
    ftrain_json = os.path.join(pdata_dir, 'final_train.json')
    filter_json_by_duration(input_json_path=ptrain_json, 
                                output_json_path=ftrain_json)

    # TESTING DATA -------------------------
    # extracting metadata of master files
    print('- PROCESSING TESTING DATA -')
    eaf_manager.extract_metadata(
        data_dir=test_dir,
        output_json_path=master_test_json,
        transcription_ext='eaf',
        audio_ext='wav',
    )
    # segment eaf and audio and generates their metadata including transcription
    eaf_manager.make_data(
        input_json_file=master_test_json,
        output_json_file=pdata_dir+'/test.json',
        output_dir=ptest_dir,
        input_audio_ext='wav',
        output_audio_ext='wav'
    )

    # FILTERING 
    # excluding data with duration outside 3000ms and 7000ms
    ptest_json = os.path.join(pdata_dir, 'test.json')
    ftest_json = os.path.join(pdata_dir, 'final_test.json')
    filter_json_by_duration(input_json_path=ptest_json, 
                                output_json_path=ftest_json)
    
    # FINAL STEP --------------------------------------------------
    # CREATE VOCABULARY
    train_vocab = 'train_vocab.json'
    test_vocab = 'test_vocab.json'

    # train vocab
    create_vocab(ptrain_json, train_vocab)
    # test vocab
    create_vocab(ptest_json, test_vocab)
    # merge vocabs
    merge_vocab_files(train_vocab, test_vocab, vocab)
    # removing train and test vocabs as they won't be used
    os.remove(train_vocab)
    os.remove(test_vocab)

def filter_json_by_duration(input_json_path, output_json_path, min_duration=1500, max_duration=5000):
    print('Filtrando datos por duracion (ms)')
    # Cargar el JSON desde el archivo
    with open(input_json_path, 'r') as infile:
        data = json.load(infile)

    # Filtrar los registros cuyo duration_ms esté entre min_duration y max_duration
    filtered_data = [entry for entry in data if min_duration <= entry['duration_ms'] <= max_duration]

    # Guardar el JSON filtrado en el archivo de salida
    with open(output_json_path, 'w') as outfile:
        json.dump(filtered_data, outfile, indent=4)

    print(f"Filtrado completado. Se guardó el archivo filtrado en: {output_json_path}")

def create_vocab(json_file, vocab_file):
    # Step 1: Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Step 2: Access 'transcription' column and join all into a single string
    transcriptions = ''.join(entry['transcription'] for entry in data)
    
    # Step 3: Remove spaces and get unique characters
    characters = sorted(set(transcriptions.replace(' ', '')))
    
    # Step 4: Create a dictionary with characters and their respective index
    vocab_dict = {char: idx for idx, char in enumerate(characters)}
    
    # Step 5: Dump the dictionary into vocab_file as a JSON
    with open(vocab_file, 'w') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=4)

def merge_vocab_files(vocab_file1, vocab_file2, merged_vocab_file):
    # Step 1: Load both vocabulary files
    with open(vocab_file1, 'r') as f1, open(vocab_file2, 'r') as f2:
        vocab1 = json.load(f1)
        vocab2 = json.load(f2)
    
    # Step 2: Combine the keys from both dictionaries
    combined_characters = sorted(set(vocab1.keys()).union(set(vocab2.keys())))
    
    # Step 3: Create a new dictionary with reindexed characters
    merged_vocab_dict = {char: idx for idx, char in enumerate(combined_characters)}
    
    # Step 4: Add special tokens at the end
    special_tokens = ['', '|', '[UNK]','[PAD]']
    for token in special_tokens:
        merged_vocab_dict[token] = len(merged_vocab_dict)
    
    # Step 5: Write the merged vocabulary dictionary to the merged_vocab_file
    with open(merged_vocab_file, 'w') as f_out:
        json.dump(merged_vocab_dict, f_out, ensure_ascii=False, indent=4)

def main():
    # Directories 
    dirs = utils.dirs

    data_dir = dirs['data_dir']
    pdata_dir = dirs['processed_data_dir']
    vocab = dirs['vocab']

    # Prepare data
    prepare_data(data_dir, pdata_dir, vocab)


if __name__ == '__main__':
    main()