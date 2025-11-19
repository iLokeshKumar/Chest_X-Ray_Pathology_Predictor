# 🩺 Chest X-Ray Pathology Predictor

AI-powered web application for analyzing chest X-ray images and predicting various pathologies using deep learning.

## Features

- **Multi-pathology Detection**: Predicts multiple chest conditions from X-ray images
- **Explainable AI**: Generates Grad-CAM heatmaps showing which regions influenced predictions
- **Multiple Format Support**: Accepts JPG, PNG, and DICOM (.dcm) files
- **REST API**: Programmatic access for integration with other systems
- **Privacy-Compliant**: Automatically de-identifies DICOM files

## Tech Stack

- **Backend**: Flask (Python)
- **Deep Learning**: PyTorch, torchxrayvision
- **Image Processing**: OpenCV, Pillow
- **Model**: DenseNet121 pretrained on chest X-ray datasets

## Installation

### Prerequisites

- Python 3.12 (required for library compatibility)
- pip

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
python myenvironment/app.py
```

The application will be available at `http://localhost:5000`

### Web Interface

1. Navigate to `http://localhost:5000`
2. Upload a chest X-ray image (JPG, PNG, or DICOM)
3. Click "Analyze Image"
4. View predictions and visual explanations

### API Usage

**Endpoint**: `POST /api/predict`

**Example**:
```bash
curl -X POST -F "file=@xray.jpg" http://localhost:5000/api/predict
```

**Response**:
```json
{
  "success": true,
  "predictions": {
    "Pneumonia": 0.8234,
    "Edema": 0.1234,
    ...
  },
  "timing": {
    "preprocess_ms": 45.2,
    "inference_ms": 123.4,
    "total_ms": 168.6
  },
  "device": "cuda:0"
}
```

## Project Structure

```
.
├── myenvironment/
│   ├── app.py                 # Flask application
│   ├── model_handler.py       # Model loading and inference
│   ├── image_processor.py     # Image preprocessing
│   ├── explainability.py      # Grad-CAM implementation
│   ├── templates/             # HTML templates
│   └── static/                # CSS, JS files
├── requirements.txt           # Python dependencies
└── README.md
```

## Important Notes

⚠️ **Medical Disclaimer**: This tool is for educational and research purposes only. It should NOT be used for actual medical diagnosis. Always consult qualified healthcare professionals.

⚠️ **Model Specificity**: The model is trained specifically for chest X-rays. Uploading other types of images will produce unreliable results.

## GPU Support

The application automatically uses GPU (CUDA) if available, otherwise falls back to CPU. For faster inference, ensure you have:
- NVIDIA GPU
- CUDA toolkit installed
- PyTorch with CUDA support

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Model: [torchxrayvision](https://github.com/mlmed/torchxrayvision)
- DenseNet architecture pretrained on multiple chest X-ray datasets
