# SatelliteVision AI

## Efficient Fine-Tuning of Deep Learning Models for Satellite Image Classification

SatelliteVision AI is a research demonstration developed using Python, PyTorch, and Streamlit.

The project investigates how efficiently a pretrained ResNet18 model can be adapted to geographically unseen satellite imagery using a limited number of labelled training samples.

## Classification Categories

The model classifies satellite image patches into three land-cover categories:

- **Forest:** Tree cover and forest land.
- **Residential:** Built-up areas containing buildings and infrastructure.
- **SeaLake:** Permanent water bodies, including lakes.

## Model Information

- **Architecture:** ResNet18
- **Starting Model:** EuroSAT-pretrained ResNet18
- **Fine-tuning Strategy:** Full fine-tuning
- **Training Samples:** 105 patches (35 per class)
- **Input Size:** 64 × 64 pixels
- **Spectral Bands:** B04, B03, B02
- **Number of Classes:** 3
- **Framework:** PyTorch
- **Web Application:** Streamlit

## Research Evaluation

The model was evaluated using geographically separate satellite imagery.

| Evaluation Dataset | Accuracy | Macro F1 |
|---|---:|---:|
| Wuhan September 21 | 96.59% | 93.47% |
| Wuhan September 26 | 91.07% | 86.03% |
| Hangzhou External | 96.57% | 95.81% |

These are recorded research evaluation results, not live measurements from uploaded files.

## How to Run the Application

1. Install Python.
2. Install the required packages:

   `pip install -r requirements.txt`

3. Start the application:

   `streamlit run app.py`

4. Open the local URL displayed in the terminal.

## Input Format

The application accepts NumPy `.npy` files containing raw three-band satellite image patches.

Supported shapes:

- `(3, 64, 64)` for one patch.
- `(N, 3, 64, 64)` for multiple patches.

**Note:** The application is designed for satellite image patches, not ordinary RGB photographs or complete satellite scenes.

## Project Purpose

This application provides an interactive demonstration of the thesis research on fine-tuning efficiency, classification performance, and geographic generalization.


