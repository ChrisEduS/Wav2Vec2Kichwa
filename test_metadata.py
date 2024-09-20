import os
import json
from moviepy.editor import AudioFileClip
import pympi
import warnings
warnings.simplefilter("ignore")

def extract_metadata(data_dir, transcription_ext, audio_ext, json_path, is_master=True):
    '''
        Extract metadata of data_dir. The function recursively explores directories
        until it finds two files (.eaf, .audio_ext) in the same directory.
        When is_master is False, it extracts the transcription from the "default" tier in the .eaf file,
        assuming there is only one annotation.
    '''
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

                # Si is_master es False, extraemos la transcripción del tier "default"
                if not is_master:
                    eaf_obj = pympi.Elan.Eaf(eaf_file)
                    
                    # Asumimos que solo hay una anotación en el tier "default"
                    transcription = eaf_obj.get_annotation_data_for_tier("default")[0][2] if "default" in eaf_obj.get_tier_names() else "No default tier found"
                    
                    # Añadimos la transcripción al registro
                    record['transcription'] = transcription

                # Añadimos el registro a la lista de resultados
                result.append(record)

                # Reseteamos las variables para buscar nuevos pares
                eaf_file = None
                audio_file = None

    # Comienza la búsqueda recursiva
    explore_directory(data_dir)

    # Guardamos el resultado en un archivo JSON
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=4)



extract_metadata(
    data_dir='/root/Wav2Vec2Kichwa/segmented_data/train',
    json_path='/root/Wav2Vec2Kichwa/segmented_data/train.json',
    audio_ext='wav',
    transcription_ext='eaf',
    is_master=False
)


# with open('/root/Wav2Vec2Kichwa/segmented_data/train.json', 'r') as f:
#     data = json.load(f)

# print(data[0])