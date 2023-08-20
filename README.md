# Comparative Analysis of Text Summarization Models

This repository contains the code and experiments for a comparative analysis of various models used for automatic text summarization. The research aims to evaluate the performance of transformer-based models, classical algorithms, and large language models (LLMs) on different datasets.

## Table of Contents
- [Introduction](#introduction)
- [Models Used](#models-used)
- [Code Structure](#code-structure)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction
Automatic text summarization is a challenging task in the field of Natural Language Processing (NLP). This research aims to provide insights into the performance of various models on this task, using different evaluation metrics such as ROUGE and BLEU.

## Models Used
1. **Transformers**:
   - BART
   - BERT
   - T5
   - Pegasus

2. **Classical Algorithms**:
   - Latent Semantic Analysis (LSA)
   - TextRank
   - SumBasic
  
3. **Large Language Models (LLMs)**:
   - GPT-2

## Code Structure
The provided code sample is for the BART model. Similar code structures are used for other models.

- **Initialization**: The code begins by importing necessary libraries and setting up the environment.
- **Model Loading**: The BART model and tokenizer are loaded using the HuggingFace transformers library.
- **Summarization Function**: A function `generate_summary` is defined to produce summaries for given texts.
- **Evaluation Metrics**: Functions to compute ROUGE and BLEU scores are provided.
- **Dataset Processing**: The code supports multiple datasets like `samsum`, `cnn_dailymail`, `xsum`, etc. These datasets are loaded, processed, and summaries are generated for them.
- **Results**: The generated summaries are evaluated against reference summaries, and the results are displayed and saved as CSV files.

## Results
The results of the experiments are saved in CSV format. For the BART model, the results are saved in `summaries_bart.csv`.

## Usage
1. Clone the repository.
2. Install the required dependencies.
3. Run the code for each model.
4. Analyze the results in the generated CSV files.

## Dependencies
- Python 3.x
- PyTorch
- HuggingFace Transformers
- Datasets
- Rouge
- NLTK
- Pandas

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
