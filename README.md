# Human-centered Explainable Psychiatric Disorder Diagnosis System using Wearable ECG Monitors
This repository contains the code for the paper "Heart2Mind: Human-Centered Contestable Psychiatric Disorder
Diagnosis System using Wearable ECG Monitors".

## Cardiac Monitoring Interface (CMI)
CMI is a web-based application that allows users to stream and visualize wearable ECG data in real-time. It provides a user-friendly interface for monitoring heart rate and other cardiac parameters.
### Installation
1. Clone the repository:
- (Optional) Create a separate virtual environment.
```
git clone https://github.com/Analytics-Everywhere-Lab/heart2mind_CMI.git
cd heart2mind_CMI
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Run the application:
```
python main.py
```
4. Open your web browser and navigate to `http://localhost:8050` to access the CMI interface.

## Contestable Diagnosis Interface (CDI)
- Install the required dependencies:
```
pip install -r requirements.txt
```
### Multi-Scale Temporal-Frequency Transform (MSTFT)
MSTFT is a transformer-based model designed for psychiatric disorder diagnosis using ECG data. It leverages multi-scale temporal and frequency information to improve classification performance.
1. (Optional) Create a separate virtual environment.
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Train the model:
```
cd models/
python train_mstft.py
```
Other models can be trained similarly by running their respective training scripts. The best checkpoint will be saved in the `checkpoints` directory.

### Contestable LLMs
Contestable LLMs are large language models designed to provide explanations and justifications for their predictions. They can be used to enhance the interpretability of the MSTFT model's predictions.
1. Run the interface:
```
python app.py
```
2. Open your web browser and navigate to `http://localhost:7860` to access the CDI interface.

## Citation
If you find this work useful, please consider citing our paper:
```
```