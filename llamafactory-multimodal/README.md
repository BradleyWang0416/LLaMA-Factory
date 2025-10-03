# Multimodal Data Generation Project

This project is designed to facilitate the generation of multimodal data, specifically focusing on converting video and text descriptions into skeleton sequences using VQVAE (Vector Quantized Variational Autoencoder) models. The project is structured to allow for easy integration of different prompt templates and VQVAE models while sharing the same dataset.

## Project Structure

- **src/llamafactory/extras_byBrad/**: Contains the main source code for the project.
  - **generate_multimodal_data.py**: Orchestrates the overall process of generating multimodal data by importing necessary modules for reading datasets, converting VQVAE data, and generating JSONL files based on prompt templates.
  - **read_dataset.py**: Handles loading raw data from specified directories and prepares it for further processing.
  - **vqvae_convert.py**: Manages the encoding process of joint data into VQVAE codebook indices.
  - **save_numpy_files.py**: Responsible for saving processed data as NumPy files, ensuring proper organization of output directories.
  - **prompt_templates.py**: Defines various prompt templates used for generating skeleton sequences from videos or text descriptions.
  - **__init__.py**: Marks the directory as a Python package.
  - **utils.py**: Contains utility functions used across different modules in the project.

- **data/_multimodal_data_byBrad/**: Directory for storing datasets and processed files.
  - **raw_dataset/**: Intended for storing raw dataset files that will be processed.
  - **vqvae_codebook/**: Designated for storing VQVAE codebook files generated during the conversion process.
  - **numpy_files/**: Where processed NumPy files will be saved after reading the dataset and converting the data.

## Setup Instructions

1. **Clone the Repository**: 
   ```bash
   git clone <repository-url>
   cd llamafactory-multimodal
   ```

2. **Install Dependencies**: 
   Ensure you have Python 3.x installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**: 
   Place your raw dataset files in the `data/_multimodal_data_byBrad/raw_dataset/` directory.

4. **Run the Data Generation Process**: 
   Execute the main script to generate multimodal data:
   ```bash
   python src/llamafactory/extras_byBrad/generate_multimodal_data.py --args
   ```

## Usage Guidelines

- Modify the prompt templates in `prompt_templates.py` to customize the skeleton generation prompts.
- Adjust the VQVAE model parameters in `vqvae_convert.py` to experiment with different configurations.
- Use `save_numpy_files.py` to manage how and where the processed data is saved.

## Overview

This project aims to bridge the gap between video analysis and motion representation through skeleton sequences, leveraging advanced machine learning techniques. By modularizing the codebase, it allows for flexibility and scalability in handling multimodal data generation tasks.