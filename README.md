# Qwen OCR for Invoices

Invoice data extraction system using Qwen2.5-VL with automatic CUIT detection and provider-specific templates.

## 🎯 Features

- **PDF Processing**: Automatic PDF to image conversion
- **CUIT Detection**: Automatic provider identification by CUIT
- **Specific Templates**: Custom configuration per provider
- **JSON Extraction**: Structured output with specific fields
- **Batch Processing**: Multiple invoice processing

## 📋 Fields extracted

The system extracts provider-specific fields based on CUIT detection:
- CAE (Electronic Authorization Code)
- CUIT (Tax ID)
- Company Name
- Total amounts and taxes
- Provider-specific tax fields (VAT, IIBB, etc.)
- Custom fields per provider configuration

## 🔧 Installation

```bash
python -m venv qwen_env
.\qwen_env\Scripts\activate
pip install requests huggingface_hub "accelerate>=0.26.0" pillow ipykernel jupyter
pip install git+https://github.com/huggingface/transformers
pip install qwen-vl-utils
pip install PyMuPDF  # For PDF processing
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📖 Usage

### Individual processing
```bash
python qwen_invoice_processor.py
```

### Batch processing
```python
from qwen_invoice_processor import QwenInvoiceProcessor

processor = QwenInvoiceProcessor()
results = processor.process_batch("./invoices/", "results.json")
```

### Configure new provider
```python
# In invoice_templates.py
PROVIDER_TEMPLATES["30123456789"] = InvoiceTemplate(
    cuit="30123456789",
    provider_name="NEW PROVIDER S.A.",
    prompt="...",  # Specific prompt
    json_schema={...}  # Fields to extract
)
```

## 📁 Project structure

```
qwen_ocr_test/
├── qwen_invoice_processor.py    # Main processor
├── invoice_templates.py         # Provider templates
├── qwen_original_params.py      # Original script
├── test_invoices/              # Test invoices (excluded from git)
├── invoice_results.json        # Results (excluded from git)
└── README.md
```

## 🔒 Security

The `.gitignore` is configured to exclude:
- PDF invoice files
- `test_invoices/` directory
- JSON results with extracted data
- Temporary images

## ⚙️ Requirements

- Python 3.8+
- CUDA GPU (6GB+ recommended)
- Windows
- PyMuPDF for PDFs

## 🎨 Available scripts

- `qwen_invoice_processor.py`: Complete system with specific templates
- `qwen_original_params.py`: Basic original script
- `qwen_simple_test.py`: Simple tests