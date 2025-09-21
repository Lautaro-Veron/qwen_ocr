# Qwen OCR for Invoices

Invoice OCR using Qwen2.5-VL.

## Installation

```bash
python -m venv qwen_env
.\qwen_env\Scripts\activate
pip install requests huggingface_hub "accelerate>=0.26.0" pillow ipykernel jupyter
pip install git+https://github.com/huggingface/transformers
pip install qwen-vl-utils
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

```bash
python qwen_original_params.py
```

## Requirements

- Python 3.8+
- CUDA GPU (6GB+ recommended)
- Windows