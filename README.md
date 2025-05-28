# KgSum

**Thesis Project** for Bachelor's Degree  
**University of Salerno**  
Lab: ISISLab  
Author: Mario Cosenza  
Supervisor: Maria Angela Pellegrino  

## Description

KgSum is a Python application for extracting, preparing, and classifying Knowledge Graphs (KGs). It combines Large Language Models (Mistral Instructor 7B with QLoRA) and traditional machine learning methods (SVM, Random Forest) to build a semantic feature extraction and classification pipeline.

## Requirements

- Python 3.12  
- CUDA 12.8  
- NVIDIA GPU (recommended: RTX 3070 or higher)  

Install dependencies:  
```bash
pip install -r requirements.txt
```

## Environment Variables

- `GEMINI_API_KEY`: API key for Gemini models, required for data extraction
- `LOCAL_ENDPOINT_LOV`: URL of the local SPARQL endpoint for LOV (recommended to run locally)

## Data Extraction Workflow

1. Download the latest JSON snapshot from the LOD Cloud
2. Set `GEMINI_API_KEY` in your environment
3. Run the scripts in order:

```bash
python endpoint_lod_service.py
python github_search.py
python zenodo_records_extraction.py
```

**Note:** Ensure you do not exceed the API rate limits.

## Data Preparation

Prepare local datasets:

```bash
python data_preparation.py
python data_preparation_remote.py
```

(Optional) Include LOV tags and comments:

```bash
python lov_data_preparation.py
```

Make sure `LOCAL_ENDPOINT_LOV` points to your SPARQL endpoint.

## Utility & Preprocessing

Run the utility and preprocessing steps:

```bash
python util.py
python preprocessing.py
```

## Pipeline Build & Training

1. Open `pipeline_build.py`
2. Specify in the code:
   - Classifier type (LLM or traditional ML)
   - Features to use
3. Execute:

```bash
python pipeline_build.py
```

## Running the Application

Start the web service with the trained model:

```bash
python app.py
```

## API Usage

Send POST requests to:
- `/api/v1/profile/sparql`
- `/api/v1/profile/file`

Refer to the Swagger documentation (coming soon) for request and response formats.

## Hardware Requirements

| Component | Minimum Specification |
|-----------|----------------------|
| GPU | NVIDIA GPU with 8+ GB VRAM (RTX 3070 or equivalent) |
| RAM | 32 GB |
| CPU | Modern multi-core processor |
| CUDA | 12.8 |

