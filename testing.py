import os
import json
from moviepy.editor import AudioFileClip
from pydub import AudioSegment
import pympi
import shutil
import subprocess

def segment_eaf_audio(audio_path, eaf_path, input_audio_ext, output_audio_ext, output_dir, output_json_path):
    seg_file_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Make output directory, no problem if it already exists
    os.makedirs(output_dir, exist_ok=True)

    # Load .eaf and audio file
    eaf = pympi.Elan.Eaf(eaf_path)

    # Use pydub with custom buffer size
    audio = AudioSegment.from_file(audio_path, format=input_audio_ext)

    # Get frequency sampling (assuming it's 44100 Hz for example)
    fs = audio.frame_rate

    # Prepare a list to store metadata
    metadata_list = []

    # Iterating over `default` tier
    tier_name = 'default'
    annotations = eaf.get_annotation_data_for_tier(tier_name)

    for idx, annotation in enumerate(annotations):
        start_time, end_time, label = annotation
        segment = audio[start_time:end_time]

        # Create directories for each output
        segment_dir = os.path.join(output_dir, seg_file_name + f'_s{idx}')
        os.makedirs(segment_dir, exist_ok=True)

        # Define the output paths
        audio_output_path = os.path.join(segment_dir, f'{seg_file_name}_s{idx}.{output_audio_ext}')
        eaf_output_path = os.path.join(segment_dir, f'{seg_file_name}_s{idx}.eaf')

        # Export the audio segment with the specified format
        segment.export(audio_output_path, format=output_audio_ext)

        # Create a new .eaf file for this segment
        new_eaf = pympi.Elan.Eaf()

        # Copy the current annotation into the new EAF file
        new_eaf.add_tier(tier_name)
        new_eaf.add_annotation(tier_name, 0, end_time - start_time, label)

        # Save the new .eaf file
        new_eaf.to_file(eaf_output_path)

        # Calculate duration in milliseconds
        duration_ms = len(segment)

        # Create metadata dictionary
        metadata = {
            'eaf_path': eaf_output_path,
            'audio_path': audio_output_path,
            'duration_ms': duration_ms,
            'fs': fs,
            'transcription': label
        }

        # Append metadata to the list
        metadata_list.append(metadata)

    # Write metadata to a JSON file
    # Check if the output JSON file exists
    if os.path.exists(output_json_path):
        # Load existing data
        with open(output_json_path, 'r') as json_file:
            existing_data = json.load(json_file)
        existing_data.extend(metadata_list)  # Add new metadata to existing data

        # Write updated data back to the JSON file
        with open(output_json_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)
    else:
        # Create a new JSON file if it does not exist
        with open(output_json_path, 'w') as json_file:
            json.dump(metadata_list, json_file, indent=4)



def make_data(input_json_file, output_dir, output_json_file, input_audio_ext, output_audio_ext):
    ''' Create segmented_data directory
    This function uses segment_eaf_audio function to segment all audio and eaf files
    It ensures to process once all files, if the output dir already exists the function does nothing
    Works in that way to save resources each time the experiment runs
    json_file is the metadata containing (among other stuff) audio and eaf files path
    '''
    #Check if directory exists
    if os.path.isdir(output_dir):
        print(f'Data is already segmented. {output_dir} already exists')
        return
    # Create output directory
    os.makedirs(output_dir)
    # Open json file 
    with open(input_json_file) as f:
        metadata = json.load(f)
    # Iter json for sample files and segment them
    for sample in metadata:
        eaf_path = sample['eaf_path']
        audio_path = sample['audio_path']
        segment_eaf_audio(
            audio_path=audio_path,
            eaf_path=eaf_path,
            input_audio_ext=input_audio_ext,
            output_audio_ext=output_audio_ext,
            output_dir=output_dir,
            output_json_path=output_json_file
        )

make_data(input_json_file='/root/Wav2Vec2Kichwa/data/test.json',
          output_json_file='/root/Wav2Vec2Kichwa/segmented_data/test.json',
          output_dir='/root/Wav2Vec2Kichwa/segmented_data/test',
          input_audio_ext='wav',
          output_audio_ext='wav')