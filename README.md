# ChatCVD: A Retrieval-Augmented Chatbot for Personalized Cardiovascular Risk Assessment with a Comparison of Medical-Specific and General-Purpose LLMs

## Overview
This repository contains the code and models introduced in our paper:  
> **"ChatCVD: A Retrieval-Augmented Chatbot for Personalized Cardiovascular Risk Assessment with a Comparison of Medical-Specific and General-Purpose LLMs"**  
> *Lakhdhar, Wafa, Maryam Arabi, Ahmed Ibrahim, Abdulrahman Arabi, and Ahmed Serag. AI 6, no. 8 (2025)*
> *[Downlad Paper](https://www.mdpi.com/2673-2688/6/8/163)*

This project conducts a comparative analysis of specialized medical language models against general-purpose language models. The objective is to evaluate their effectiveness in handling medical information and tasks. Following this analysis, we implement Retrieval-Augmented Generation (RAG) to enhance recommendation generation based on the findings.

## Table of Contents
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [BioBERT Example](#biobert-example)
- [Contributing](#contributing)

## Installation
To set up the project, install the required packages using pip. You can create a virtual environment and run:

```bash
pip install -r requirements.txt
```

## Data Preprocessing
The preprocessing script (`preprocess.py`) handles:
- Loading raw data from CSV files.
- Cleaning and transforming data.
- Mapping categorical variables to text descriptions.

To run the preprocessing script, use the following command:

```bash
python preprocess.py --input /path/to/your/data.csv --output preprocessed_data.csv
```

## BioBERT Example
To fine-tune the BioBERT model for predicting cardiovascular disease risk, run the following command:

```bash
python fine_tune.py --train /path/to/train_cvd.csv --val /path/to/val_cvd.csv --test /path/to/test_cvd.csv
```

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for discussion.
