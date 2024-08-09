# Depression Detection using BERT on Social Media Platforms

This repository contains code for detecting depression through social media posts using BERT for feature extraction and an autoencoder for dimensionality reduction.

## Overview

This project aims to identify signs of depression in text data by combining the strengths of BERT for capturing contextual information and autoencoders for reducing feature dimensionality.

## Key Features

- **BERT Model:** Utilizes a pre-trained BERT model for effective feature extraction from text data.
- **Autoencoder:** Reduces the dimensionality of extracted features to improve classification performance.
- **Data Preprocessing:** Includes steps for data cleaning, tokenization, and preparation.
- **Model Training:** Scripts for training the autoencoder and fine-tuning the BERT model.
- **Evaluation:** Performance metrics such as accuracy, precision, recall, and F1 score for model assessment.

## Requirements

- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- Scikit-learn
- Matplotlib

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/anuraag165/Depression_BERT_Autoencoder.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Depression_BERT_Autoencoder
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your dataset and place it in the `data/` directory. The dataset should include:
    - `depression_dataset_reddit_cleaned.csv`: Original data retrieved from Kaggle.
    - `merged_tensors_with_labels.csv`: Encoded data using BERT.
2. Run the training script located in the `py/` directory:
    ```bash
    python py/bert_v_autoencoder.py
    ```

## Google Colab Integration

To run the code in Google Colab:

1. Open the notebook file in the `ipynb/` folder:
    - `Colab_Notebook.ipynb`

2. Follow these steps to set up the notebook in Colab:
    - Upload the dataset files (`depression_dataset_reddit_cleaned.csv` and `merged_tensors_with_labels.csv`) to the Colab environment.
    - Ensure all required libraries are installed in the Colab environment by running:
      ```python
      !pip install torch transformers scikit-learn matplotlib
      ```
    - Run the cells in the notebook to preprocess data, train models, and evaluate performance.

## Datasets

This project uses the following datasets:
- **`depression_dataset_reddit_cleaned.csv`**: Original dataset retrieved from Kaggle.
- **`merged_tensors_with_labels.csv`**: Encoded data using BERT.

## Acknowledgments

This project is built upon the work of researchers and developers in the fields of NLP and deep learning, particularly those who developed BERT and autoencoders.

## License

This project is licensed under the MIT License.

## Programmer

- Anuraag Raj
