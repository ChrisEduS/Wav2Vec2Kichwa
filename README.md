# Guia de Ejecución

## Ubicación de los datos
### Datasets Originales
Los datos se encuentran en el servidor DGX, en el usuario csantamarial, la ruta completa es `/home/csantamarial/org_data`, aquí se pueden encontrar los dos datasets `csantamaria` (creado conjuntamente con el voluntario pagado), y `killkan` (obtenido del repositorio `https://github.com/ctaguchi/killkan`).
killkan ya tiene hechas las divisiones de audio en el repositorio general, sin embargo hay casos en donde hay audios sin transcripciones o transcripciones sin audios, por lo que se optó por tomar únicamente los audios y transcripciones maestros. Para ello, existe el archivo `extract_masters.py`, al cual se le deben cambiar las rutas de killkan y el directorio destino. 
csantamaria tiene todos los archivos maestros: audios maestros, transcripciones y revisiones corregidas. El dataset limpio se obtuvo de manera manual. 
### Datasets Limpios
Los datasets limpios se encuentran en `/home/csantamarial/processed_master_data`, aquí tenemos los datasets test y train. Test es el dataset `csantamaria` y train es `killkan`. 
Este dataset ya puede ser ingresado en el contenedor de docker para utilizarlo. 

## Configuraciones de rutas
El archivo `utils.py` contiene los diccionarios de las direcciones en donde se encuentran los datos. Se puede cambiar de acuerdo a la ubicación en donde se pongan los datos. 
`data_dir` path en donde estarán los datos maestros limpios.
`processed_data_dir` path en donde se almacenarán los datos segmentados. 
`vocab` ruta del archivo `.json` en donde se guardará el vocabulario para el modelo. 
`checkpoints` path en donde se guardarán los checkpoints del entrenamiento.

## Preprocesamiento de datos
El modelo no puede procesar audios extensos por lo que es necesario que estos sean segmentados en audios más cortos. Para ello, se debe ejecutar el archivo `/prep_files/data_prep.py`, usando el comando `python3 data_prep.py`. 
Lo único que se puede alterar en este archivo es la función `filter_json_by_duration` en donde se define la extensión mínima y máxima que pueden tener los audios segmentados, en milisegundos. 

## Experimentaciones

