import torch
import utils
from datamodule import DataModule_KichwaWav2vec2 

def main():
    # Configuraciones básicas
    dirs = utils.dirs
    configs = utils.configs

    data_dir = dirs['data_dir']
    processed_data_dir = dirs['processed_data_dir']

    freq_sample = configs['fs']
    batch_size = configs['batch_size']  # Define el tamaño de lote que desees

    # Crear una instancia del DataModule
    data_module = DataModule_KichwaWav2vec2(data_dir=data_dir,
                                            processed_data_dir=processed_data_dir,
                                            freq_sample=freq_sample,
                                            batch_size=batch_size)
    
    # Preparar los datos (solo si no has procesado los datos antes)
    data_module.prepare_data()
    
    # Configurar el DataModule para el entrenamiento
    data_module.setup(stage='test')

    # Obtener el train_dataloader
    test_loader = data_module.test_dataloader()
    dataset = test_loader.dataset

    print(dataset[0]['audio'], dataset[0]['audio'].shape) #dataset[0]['fs']
    print('Cantidad de datos para testear:', len(test_loader.dataset))


def sum_duration_in_memory(dataset):
    durations = [data['duration'] for data in dataset]
    total_duration = sum(durations)
    return total_duration


if __name__ == "__main__":
    main()
