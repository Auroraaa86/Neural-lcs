# Neural LCS(Phoneme/Word)

## Overview
This repository contains a project that involves training and inference for phoneme and word-level alignment models. The project is structured to handle both phoneme-level and word-level data, with separate directories and models for each level.

## Project Structure

- `phn_lcs/`: Phoneme-level alignment directory.
  - `data_phn/`: Contains phoneme-level data.
  - `model/`: You should create a folder named "model" here, and you can download the pretrained phoneme aligner model weight file in the [link](https://drive.google.com/drive/folders/1u94UGi1TQfS3E4Cvd2-HqK-TE7K0IM-N) .
  - `inference.ipynb`: Jupyter notebook for phoneme-level inference.
  - `train.py`: Script for training phoneme alignment models.
- `word_lcs/`: Word-level alignment directory.
  - `data_word/`: Contains word-level data.
  - `model/`: The same as phn_lcs, you can download the word aligner model weight file in the same link.
  - `inference.ipynb`: Jupyter notebook for word-level inference.
- `requirements.txt`: List of required Python dependencies.
- `simulation/`: Contains scripts for phoneme and word-level simulation.
  - `generator_phn.ipynb`: Jupyter notebook for generating phoneme-level data.
  - `generator_word.ipynb`: Jupyter notebook for generating word-level data.

## How to Use

### Phoneme-Level Inference

1. Navigate to the `phn_lcs/` directory.
2. Open the `inference.ipynb` Jupyter notebook.
3. Follow the steps outlined in the notebook to perform phoneme-level alignment and inference.


### Word-Level Inference

1. Navigate to the `word_lcs/` directory.
2. Open the `inference.ipynb` Jupyter notebook.
3. Follow the steps outlined in the notebook to perform word-level alignment and inference.

### Training a Model

To train a new model for phoneme alignment:
1. Run the `train.py` script.
2. The script will use the data in `data_phn/` to train the model.

## Requirements

Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.