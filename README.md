# Low-Rank Autoencoder for CAN Bus Intrusion Detection

This repository implements the **Low-Rank Autoencoder (LRAE)** for anomaly detection in **Controller Area Network (CAN) bus** systems. It demonstrates significant improvements in efficiency and performance compared to the traditional **Standard Autoencoder (SAE)**.

## Key Contributions:
- 91.3% reduction in model parameters (2,010 vs 23,198).
- 9× lower memory footprint (0.01 MB vs 0.09 MB).
- Faster training convergence: 50% faster than SAE.
- Enhanced anomaly detection with significant improvements in precision, recall, and F1-score across multiple test scenarios.

## Requirements
- Python 3.6+
- PyTorch
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/LowRankAutoencoder_CAN_Bus_Intrusion_Detection.git
cd LowRankAutoencoder_CAN_Bus_Intrusion_Detection
pip install -r requirements.txt
