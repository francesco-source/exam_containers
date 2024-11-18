import os
from pathlib import Path
import yaml
from hashlib import sha512
import sys
import numpy as np
from datasets import load_dataset, DownloadConfig
from transformers import CLIPTokenizerFast, CLIPProcessor,TFCLIPModel
from tqdm.auto import tqdm
import pickle

import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  

import tensorflow as tf

logging.getLogger().setLevel(logging.ERROR) 

tf.get_logger().setLevel('ERROR') 

def get_env_param(name, default):
    """Helper function to get environment parameters with defaults"""
    return os.getenv(name, default)

# Set up environment parameters with safe defaults
DATASET_NAME = get_env_param('DATASET_NAME', "frgfm/imagenette")
DATASET_CONFIG = get_env_param('DATASET_CONFIG', "full_size")
DATASET_SPLIT = get_env_param('DATASET_SPLIT', "train")
MODEL_ID = get_env_param('MODEL_ID', "openai/clip-vit-base-patch32")
SAMPLE_SIZE = int(get_env_param('SAMPLE_SIZE', "100"))
BATCH_SIZE = int(get_env_param('BATCH_SIZE', "16"))
PROMPT = get_env_param('PROMPT', "a dog in the snow")
TOP_K = int(get_env_param('TOP_K', "5"))
RANDOM_SEED = int(get_env_param('RANDOM_SEED', "0"))

# Compute experiment ID based on parameters
FOOTPRINT_KEYS = {
    'DATASET_NAME', 'DATASET_CONFIG', 'DATASET_SPLIT', 'MODEL_ID',
    'SAMPLE_SIZE', 'BATCH_SIZE', 'PROMPT', 'TOP_K', 'RANDOM_SEED'
}
EXPERIMENT_FOOTPRINT = {k: locals()[k] for k in FOOTPRINT_KEYS}
EXPERIMENT_FOOTPRINT_YAML = yaml.dump(EXPERIMENT_FOOTPRINT, sort_keys=True)
EXPERIMENT_ID = sha512(EXPERIMENT_FOOTPRINT_YAML.encode()).hexdigest()

# Setup directories
DATA_DIR = Path(get_env_param('DATA_DIR', './data'))
OUTPUT_DIR = Path(get_env_param('OUTPUT_DIR', str(DATA_DIR)))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create images directory for this experiment
IMAGES_DIR = OUTPUT_DIR / 'images' / EXPERIMENT_ID[:8]
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Check if experiment already exists
for file in DATA_DIR.glob('**/*.yaml'):
    if file.is_file() and file.stem == EXPERIMENT_ID:
        print(f'Experiment with ID {EXPERIMENT_ID[:8]} already exists in {file.parent}', file=sys.stderr)
        exit(0)

print(f'Running experiment ID {EXPERIMENT_ID[:8]}', file=sys.stderr)
print(f'Output directory: {OUTPUT_DIR}', file=sys.stderr)
print(f'Experiment footprint:\n\t{EXPERIMENT_FOOTPRINT_YAML.replace("\n", "\n\t")}', file=sys.stderr)

# Set device
device = "cpu"
print(f"Using device: {device}")

# Load dataset
dataset = load_dataset(
    DATASET_NAME,
    DATASET_CONFIG,
    split=DATASET_SPLIT,
)

print("Done")

# Initialize model and processors
model = TFCLIPModel.from_pretrained(MODEL_ID)
tokenizer = CLIPTokenizerFast.from_pretrained(MODEL_ID)
processor = CLIPProcessor.from_pretrained(MODEL_ID)

# Process text prompt
inputs = tokenizer(PROMPT, return_tensors="tf")
text_emb = model.get_text_features(**inputs)

# Sample and process images
np.random.seed(RANDOM_SEED)
sampled_idx = np.random.randint(0, len(dataset) + 1, SAMPLE_SIZE).tolist()
images = [dataset[i]["image"] for i in sampled_idx]

# Process images in batches
image_arr = None
for i in tqdm(range(0, len(images), BATCH_SIZE)):
    batch = images[i:i + BATCH_SIZE]
    batch = processor(
        text=None,
        images=batch,
        return_tensors="tf",
        padding=True
    ).pixel_values
    
    batch_emb = model.get_image_features(pixel_values=batch)
    batch_emb = batch_emb.numpy()
    
    if image_arr is None:
        image_arr = batch_emb
    else:
        image_arr = np.concatenate((image_arr, batch_emb), axis=0)

# Normalize embeddings
image_arr = image_arr.T / np.linalg.norm(image_arr, axis=1)
image_arr = image_arr.T
text_emb = text_emb.numpy()

# Calculate similarity scores
scores = np.dot(text_emb, image_arr.T)

# Get top K matches
top_indices = np.argsort(-scores[0])[:TOP_K]

# Save top K images
image_paths = []
for idx, i in enumerate(top_indices):
    image_path = IMAGES_DIR / f'top_{idx+1}_score_{scores[0][i]:.3f}.png'
    images[i].save(image_path)
    image_paths.append(str(image_path))

# Save results
results = {
    'top_k_indices': top_indices.tolist(),
    'top_k_scores': scores[0][top_indices].tolist(),
    'text_embedding': text_emb.tolist(),
    'image_embeddings': image_arr.tolist(),
    'image_paths': image_paths,
    'prompt': PROMPT
}

with open(OUTPUT_DIR / 'results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Save experiment footprint
with open(OUTPUT_DIR / f'{EXPERIMENT_ID}.yaml', 'w') as f:
    yaml.dump(EXPERIMENT_FOOTPRINT, f)