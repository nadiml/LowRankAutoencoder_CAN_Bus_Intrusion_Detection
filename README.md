# LRAE: Low-Rank Autoencoder for Real-Time CAN Bus Intrusion Detection

This repository contains the implementation of the Low-Rank Autoencoder (LRAE) for real-time efficient intrusion detection in Controller Area Network (CAN) bus systems, as presented in the paper accepted at WINCOM'2025. The code is designed to demonstrate the model's efficiency and performance improvements over traditional Standard Autoencoders (SAE).

## Overview
- **Paper**: "LRAE: A Low-Rank Autoencoder for Real-Time Efficient CAN Bus Intrusion Detection" by Nadim Ahmed et al.
- **Conference**: The 12th International Conference on Wireless Networks and Mobile Communications (WINCOM'2025), Riyadh, Saudi Arabia, November 25-27, 2025.
- **Authors**: Nadim Ahmed, Md. Ashraful Babu, Md. Manir Hossain Mollah, Md. Mortuza Ahmmed, M. Mostafizur Rahman, Mufti Mahmud.

## Repository Structure
- `main.py`: Main script to run the LRAE and SAE models, including training, evaluation, and visualization.
- `requirements.txt`: List of dependencies required to run the code.
- `results/`: Directory to store output files (e.g., CSV results, plots).
- `data/`: Directory for SynCAN dataset (to be placed manually).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lrae-can-intrusion-detection.git
   cd lrae-can-intrusion-detection
