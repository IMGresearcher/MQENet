# MQENet

MQENet: A Mesh Quality Evaluation Neural Network Based on Dynamic Graph Attention.

## Project Overview

MQENet is designed to evaluate mesh quality through graph-based representations of computational meshes. The framework supports 2D structured meshes (GRD format), converting them into graph structures suitable for deep learning analysis.

## File Structure

### Core Files

- **`model.py`** - Contains the main neural network architecture
  - `DeepGCNLayer`: Custom graph convolutional layer with residual connections
  - `DeeperGCN`: Main model implementing a deep graph convolutional network
  - `TraceModel`: Wrapper for model tracing and deployment

- **`preprocess.py`** - Mesh preprocessing and graph conversion utilities
  - `grd2vertex_graph()`: Converts 2D GRD meshes to vertex-based graphs
  - `grd2element_graph()`: Converts 2D GRD meshes to element-based graphs
  - `stl2vertex_graph()`: Converts 3D STL meshes to vertex-based graphs
  - `stl2element_graph()`: Converts 3D STL meshes to element-based graphs
  - `compute_proximity_adjacency()`: Creates adjacency matrices based on spatial proximity
  - `preprocess_grd_meshes()`: Batch processing for GRD files
  - `preprocess_stl_meshes()`: Batch processing for STL files

- **`mesh_dataset.py`** - Dataset classes for mesh data loading
  - `InMemMeshDataset`: In-memory dataset for small datasets
  - `MeshDataset`: Standard dataset class for larger datasets
  - `StreamMeshDataset`: Streaming dataset for very large datasets
  - `GrdDataset`: Specialized dataset for Kaggle GRD data
  - `grd_label()`: Label generation function for GRD files
  - `stl_label()`: Label generation function for STL files

- **`utils.py`** - Training and utility functions
  - `train_model()`: Complete training loop with validation and early stopping
  - `test_model()`: Model evaluation function
  - `split_dataset()`: Dataset splitting utilities
  - `save_model()`: Model checkpointing
  - `add_weight_decay()`: Optimizer configuration

- **`torchserve.py`** - Model deployment and serving utilities
  - `generate_jit_model()`: Converts models to TorchScript for deployment
  - `pack_model()`: Creates MAR files for TorchServe deployment
  - `register_model()`: Registers models with TorchServe
  - `scale_model()`: Manages model scaling

### Training Models
```python
from mesh_dataset import InMemMeshDataset
from model import DeeperGCN
from utils import train_model

# Create dataset
dataset = MeshDataset(root='data', raw_file_root='processed', 
                          num=1024, label_func=grd_label)

# Initialize model
model = DeeperGCN(input_channels=8, num_layers=6, hidden_channels=64, 
                  num_classes=2, dropout=0.1)

# Train model
train_model(model, optimizer, criterion, train_loader, val_loader, 
           model_dict, param_file, epochs=100)
```

## Dependencies

- Python 3.8+
- PyTorch 1.12.1+ (with CUDA support recommended)
- PyTorch Geometric 2.0.4+
- NumPy, SciPy, Pandas for data processing
- TorchServe for model deployment



