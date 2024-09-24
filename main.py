import torch
import utils
from datamodule import DataModule_KichwaWav2vec2  # Asegúrate de que el nombre y la ruta del archivo son correctos

def main():
    # Configuraciones básicas
    params = utils.params
    data_dir = params['data_dir']
    processed_data_dir = params['processed_data_dir']
    batch_size = 8  # Define el tamaño de lote que desees

    # Crear una instancia del DataModule
    data_module = DataModule_KichwaWav2vec2(data_dir, processed_data_dir, batch_size)
    
    # Preparar los datos (solo si no has procesado los datos antes)
    data_module.prepare_data()
    
    # Configurar el DataModule para el entrenamiento
    data_module.setup(stage='test')

    # Obtener el train_dataloader
    test_loader = data_module.test_dataloader()
    dataset = test_loader.dataset

    print('Cantidad de datos para testear:', len(test_loader.dataset))
    print('Total de ms para testear:', sum_duration_in_memory(dataset))

def sum_duration_in_memory(dataset):
    durations = [data['duration'] for data in dataset]
    total_duration = sum(durations)
    return total_duration


if __name__ == "__main__":
    main()
