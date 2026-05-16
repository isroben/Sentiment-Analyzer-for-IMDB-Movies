import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


list_of_files = [
    'data/raw/__init__.py',
    'data/processed/__init__.py',
    'notebooks/EDA.ipynb',
    'src/__init__.py',
    'src/components/__init__.py',
    'src/components/data_ingestion.py',
    'src/components/data_transformation.py',
    'src/components/model_trainer.py',
    'src/Components/model_evaluation.py',
    'src/pipeline/__init__.py',
    'src/pipeline/train_pipeline.py',
    'src/pipeline/predict_pipeline.py',
    'src/utils/utils.py',
    'src/utils/logger.py',
    'src/utils/exception.py',
    'artifacts/__init__.py',
    'app.py',
    'templates/index.html',
    'templates/home.html',
    'setup.py',
]

for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir != '':
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) ==0):
        with open(filepath, 'w') as file:
            pass
        logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f'{filename} is already exists.')