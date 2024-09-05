# ASL Fingertyping

ASL Fingertyping is a machine learning-based project aimed at recognizing American Sign Language (ASL) alphabets and enabling typing using only finger movements. This repository contains the implementation of the ASL Fingertyping system, including data preprocessing, model training, and real-time inference.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The ASL Fingertyping project leverages machine learning techniques to recognize ASL alphabets from finger movements. The system is designed to facilitate communication for individuals who use ASL by enabling them to type words using their fingers. The project includes data collection, model training, and real-time inference components.

## Features

- **Data Collection**: Tools for collecting and preprocessing finger movement data.
- **Model Training**: Scripts for training machine learning models to recognize ASL alphabets.
- **Real-time Inference**: Real-time recognition of ASL alphabets and typing interface.

## Installation

To get started with ASL Fingertyping, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Macro1027/ASL-fingertyping.git
    cd ASL-fingertyping
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Collection

1. **Collect Data**:
    - Use the provided scripts in the `data_collection` directory to collect finger movement data.
    - Ensure that the data is saved in the appropriate format for preprocessing.

2. **Preprocess Data**:
    - Run the preprocessing scripts to clean and prepare the data for model training.
    ```bash
    python data_collection/preprocess.py --input-dir data/raw --output-dir data/processed
    ```

### Model Training

1. **Train the Model**:
    - Use the training scripts to train the machine learning model on the preprocessed data.
    ```bash
    python model_training/train.py --data-dir data/processed --model-dir models
    ```

2. **Evaluate the Model**:
    - Evaluate the trained model using the evaluation scripts.
    ```bash
    python model_training/evaluate.py --model-dir models --data-dir data/processed
    ```

### Real-time Inference

1. **Run Real-time Inference**:
    - Use the inference scripts to recognize ASL alphabets in real-time and enable typing.
    ```bash
    python inference/real_time_inference.py --model-dir models
    ```

## Project Structure

