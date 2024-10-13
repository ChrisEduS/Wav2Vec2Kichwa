import os
import re
import json
from moviepy.editor import AudioFileClip
from pydub import AudioSegment
import pympi

class EafManager:
    def __init__(self) -> None:
        pass
    def extract_metadata(self, data_dir, transcription_ext, audio_ext, output_json_path):
        '''
            Extract metadata of data_dir. The function recursively explores directories
            until it finds two files (.eaf, .audio_ext) in the same directory.
            data_dir: str - master data directory/path
            transcription_ext: str - extension of transcription files. assumed to be .eaf 
            audio_ext: str - extension of audio files
            output_json_path: str - path where metadata will be stored as json file
        '''
        print('Checking for master metadata...')
        if os.path.isfile(output_json_path):
            print(f'Master metadata already extracted. {output_json_path} already exists')
            return
        
        print('Metadata not available. Starting process...')
        result = []

        def explore_directory(current_dir):
            eaf_file = None
            audio_file = None

            # Recorremos todos los archivos y directorios dentro del directorio actual
            for entry in os.listdir(current_dir):
                full_path = os.path.join(current_dir, entry)

                # Si es un directorio, entramos recursivamente
                if os.path.isdir(full_path):
                    explore_directory(full_path)

                # Si es un archivo, verificamos su extensión
                elif os.path.isfile(full_path):
                    if entry.endswith(transcription_ext):
                        eaf_file = full_path
                    elif entry.endswith(audio_ext):
                        audio_file = full_path

                # Si encontramos ambos archivos, los registramos
                if eaf_file and audio_file:
                    # Usamos moviepy para conseguir la metadata del audio
                    audio_clip = AudioFileClip(audio_file)
                    duration_ms = int(audio_clip.duration * 1000)  # Duración en milisegundos
                    frequency_sampling = audio_clip.fps  # Frecuencia de muestreo

                    # Registro base de datos
                    record = {
                        'eaf_path': eaf_file,
                        'audio_path': audio_file,
                        'duration_ms': duration_ms,
                        'FS': frequency_sampling
                    }

                    # Añadimos el registro a la lista de resultados
                    result.append(record)

                    # Reseteamos las variables para buscar nuevos pares
                    eaf_file = None
                    audio_file = None

        # Comienza la búsqueda recursiva
        explore_directory(data_dir)

        # Guardamos el resultado en un archivo JSON
        with open(output_json_path, 'w') as f:
            json.dump(result, f, indent=4)
        print(f'Master metadata extracted. Check it at {output_json_path}')

    def segment_eaf_audio(self, audio_path, eaf_path, input_audio_ext, output_audio_ext, output_dir, output_json_path):
        '''segments the audio and eaf master files into smaller parts
        also generates metadata including transcription as column.
        audio_path: str - path of master audio file
        eaf_path: str - path of master eaf file. Transcription info.
        input_audio_ext: str - audio extension of master audio
        output_audio_ext: str - audio extension of resulting segmented audio. wav extension hardly recommended
        output_dir: str - output directory of all segmentations
        output_json_path - path of json with metadata including transcriptions as column 
        '''
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

            # cleaning label
            label = self.clean_string(label)

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



    def make_data(self, input_json_file, output_dir, output_json_file, input_audio_ext, output_audio_ext):
        ''' Create segmented_data directory
        This function uses segment_eaf_audio function to segment all audio and eaf files
        It ensures to process once all files.
        input_json_file: str - master metadata json.
        output_dir: str - output directory to store all segmentations
        output_json_file: str - segmentations metadata file path
        input_audio_ext: str - master audio extensions
        output_audio_ext: str - audio extension of resulting segmented audio. wav extension hardly recommended
        '''
        #Check if directory exists
        print('Checking for segmented data...')
        if os.path.isdir(output_dir):
            print(f'Data is already segmented. {output_dir} already exists')
            return
        print('Data not segmented yet. Starting process...')
        # Create output directory
        os.makedirs(output_dir)
        # Open json file 
        with open(input_json_file) as f:
            metadata = json.load(f)
        # Iter json for sample files and segment them
        for sample in metadata:
            eaf_path = sample['eaf_path']
            audio_path = sample['audio_path']
            self.segment_eaf_audio(
                audio_path=audio_path,
                eaf_path=eaf_path,
                input_audio_ext=input_audio_ext,
                output_audio_ext=output_audio_ext,
                output_dir=output_dir,
                output_json_path=output_json_file
            )
        print(f'Segmenting done. Check at {output_dir}. Metadata at {output_json_file}')    


    def clean_string(self, input_string: str) -> str:
        # Mapping of characters to replace
        chars_to_replace = {
            "á": "a",
            "é": "e",
            "í": "i",
            "ó": "o",
            "ú": "u",
            "ü": "u"
        }
        
        # Replace accented characters with their unaccented equivalents
        for accented_char, unaccented_char in chars_to_replace.items():
            input_string = input_string.replace(accented_char, unaccented_char)
        
        # Add numbers to the regex
        chars_to_remove_regex = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\�\' \n\[\]0-9]'
        
        # Convert the string to lowercase
        cleaned_string = input_string.lower()
        
        # Remove unwanted characters and numbers using regex
        cleaned_string = re.sub(chars_to_remove_regex, '', cleaned_string)
        
        return cleaned_string

