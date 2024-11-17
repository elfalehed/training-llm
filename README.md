# LLM Training Repository

This repository contains a modular implementation for training Language Learning Models (LLMs) using PyTorch and Hugging Face Transformers.

## Project Structure

```
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── model/          # Model training components
│   ├── utils/          # Utility functions and metrics
│   └── config/         # Configuration management
├── train.py            # Main training script
└── requirements.txt    # Project dependencies
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure training parameters in `src/config/config.py`

3. Run training:
   ```bash
   python train.py
   ```

## Features

- Modular architecture for easy customization
- Integration with Weights & Biases for experiment tracking
- Configurable training parameters
- Perplexity calculation for model evaluation
- Automatic model checkpointing

## Configuration

Modify `src/config/config.py` to adjust:
- Model selection
- Dataset choice
- Training parameters
- Output directories

## Monitoring

Training progress can be monitored through Weights & Biases dashboard.
