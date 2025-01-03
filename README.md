
# CLIP Image Retrieval System

## Overview
This project implements a scalable image retrieval system using OpenAI's CLIP (Contrastive Language-Image Pre-training) model. It allows you to search through image datasets using natural language queries and retrieve the most relevant images based on semantic similarity.

## Features
- Multiple CLIP model support (base and large variants)
- Configurable batch processing for efficient image handling
- Docker-based deployment with different experimental configurations
- Automated experiment tracking and results storage
- Flexible environment variable configuration
- Support for custom prompts and various dataset configurations

## Prerequisites
- Docker and Docker Compose
- Python 3.10 or higher (if running locally)
- At least 8GB RAM recommended
- Storage space for datasets and model weights

## Quick Start

1. Clone the repository and navigate to the project directory:
```bash
git clone <repository-url>
cd clip-retrieval-system
```

2. Build and run the experiments:
```bash
docker-compose build
docker-compose up
```

## Available Configurations

The system comes with several pre-configured experimental setups:

1. **Base Experiment** (`experiment-base`):
   - Uses CLIP ViT-Base/Patch32 model
   - Default dataset: imagenette
   - Validation split

2. **Large Model** (`experiment-large`):
   - Uses CLIP ViT-Large/Patch14 model
   - Increased batch size (32)
   - Larger sample size (200 images)

3. **Custom Prompt** (`experiment-custom-prompt`):
   - Natural landscape prompt
   - Increased top-K results (10)
   - Medium sample size (150 images)

4. **Extended Run** (`experiment-extended`):
   - Larger sample size (300 images)
   - More results (top-15)
   - Custom random seed

5. **Small Run** (`experiment-small`):
   - Minimal configuration for quick testing
   - Small batch and sample sizes
   - Portrait-focused prompt

## Environment Variables

Configure your experiments using these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_ID` | CLIP model identifier | openai/clip-vit-base-patch32 |
| `DATASET_NAME` | HuggingFace dataset name | frgfm/imagenette |
| `DATASET_CONFIG` | Dataset configuration | full_size |
| `DATASET_SPLIT` | Dataset split to use | train |
| `SAMPLE_SIZE` | Number of images to process | 100 |
| `BATCH_SIZE` | Batch size for processing | 16 |
| `PROMPT` | Text prompt for image retrieval | "a dog in the snow" |
| `TOP_K` | Number of top results to return | 5 |
| `RANDOM_SEED` | Random seed for reproducibility | 0 |

## Directory Structure

```
.
├── clip.py              # Main experiment script
├── docker-compose.yml   # Docker compose configuration
├── Dockerfile          # Docker build instructions
├── requirements.txt    # Python dependencies
└── data/              # Mount point for experiment data
    ├── images/        # Retrieved images
    └── results/       # Experiment results
```

## Output Structure

Each experiment creates a timestamped directory containing:
- Retrieved images with similarity scores
- Experiment configuration YAML
- Full output log
- Pickle file with embeddings and results

## Results Format

The system saves results in two formats:
1. Individual images in the `images` subdirectory
2. A `results.pkl` file containing:
   - Text embeddings
   - Image embeddings
   - Similarity scores
   - File paths to saved images
   - Original prompt

## Running Custom Experiments

Create a new service in `docker-compose.yml`:

```yaml
experiment-custom:
  build: *experiment_build
  volumes: *volumes
  hostname: custom
  environment:
    <<: *base_env
    MODEL_ID: "your-model-choice"
    PROMPT: "your custom prompt"
    SAMPLE_SIZE: "desired-sample-size"
```

## Development

To modify the system:
1. Update `clip.py` for core logic changes
2. Modify `docker-compose.yml` for new experimental configurations
3. Update `requirements.txt` for new dependencies

## Troubleshooting

Common issues and solutions:

1. **Out of Memory**:
   - Reduce batch size
   - Use smaller model variant
   - Decrease sample size

2. **Slow Processing**:
   - Increase batch size (if memory allows)
   - Use CPU-optimized configuration
   - Reduce dataset size

3. **Storage Issues**:
   - Clear old experiment results
   - Mount external volume
   - Reduce TOP_K value

