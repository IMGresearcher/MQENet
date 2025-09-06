# MQENet

MQENet: A Mesh Quality Evaluation Neural Network Based on Dynamic Graph Attention.

## Project Overview

MQENet is designed to evaluate mesh quality through graph-based representations of computational meshes. The framework supports both 2D structured meshes (GRD format) and 3D unstructured meshes (STL format), converting them into graph structures suitable for deep learning analysis.

## File Structure

### Core Files

- **`model.py`** - Contains the main neural network architecture
  - `DeepGCNLayer`: Custom graph convolutional layer with residual connections
  - `DeeperGCN`: Main model implementing a deep graph convolutional network
  - `TraceModel`: Wrapper for model tracing and deployment
  - Uses GATv2Conv, SAGPooling, and JumpingKnowledge for advanced graph processing

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
  - Various helper functions for model management

- **`torchserve.py`** - Model deployment and serving utilities
  - `generate_jit_model()`: Converts models to TorchScript for deployment
  - `pack_model()`: Creates MAR files for TorchServe deployment
  - `register_model()`: Registers models with TorchServe
  - `scale_model()`: Manages model scaling
  - Model management functions for production deployment

### Configuration Files

- **`requirements.txt`** - Python dependencies including:
  - PyTorch 1.12.1 with CUDA 11.6 support
  - PyTorch Geometric 2.0.4 for graph neural networks
  - Scientific computing libraries (NumPy, SciPy, Pandas)
  - Machine learning utilities (scikit-learn, optuna)

- **`__init__.py`** - Package initialization file

## Key Features

### Graph Representation
- **Vertex-based graphs**: Nodes represent mesh vertices, edges represent connectivity
- **Element-based graphs**: Nodes represent mesh elements, edges represent adjacency
- **Proximity-based adjacency**: Spatial relationships based on distance thresholds
- **Duplicate node handling**: Automatic detection and merging of duplicate vertices

### Model Architecture
- **Deep Graph Convolutional Network**: Multi-layer GCN with residual connections
- **Graph Attention**: Uses GATv2Conv for attention-based message passing
- **Graph Pooling**: SAGPooling for hierarchical graph reduction
- **Jumping Knowledge**: Aggregates features from multiple layers
- **Multi-scale features**: Combines global mean and max pooling

### Data Processing
- **2D Structured Meshes**: GRD format support with automatic graph generation
- **3D Unstructured Meshes**: STL format support with triangle-based graphs
- **Feature Engineering**: Geometric features including angles, areas, and distortions
- **Normalization**: 0-1 normalization for numerical stability

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Preprocessing Meshes
```python
from preprocess import preprocess_grd_meshes, preprocess_stl_meshes

# Process 2D GRD meshes
preprocess_grd_meshes('input_path', 'output_path', mode='E')  # Element-based
preprocess_grd_meshes('input_path', 'output_path', mode='V')  # Vertex-based

# Process 3D STL meshes
preprocess_stl_meshes('input_path', 'output_path', mode='E')  # Element-based
preprocess_stl_meshes('input_path', 'output_path', mode='V')  # Vertex-based
```

### Training Models
```python
from mesh_dataset import InMemMeshDataset
from model import DeeperGCN
from utils import train_model

# Create dataset
dataset = InMemMeshDataset(root='data', raw_file_root='processed', 
                          num=1000, label_func=grd_label)

# Initialize model
model = DeeperGCN(input_channels=8, num_layers=6, hidden_channels=64, 
                  num_classes=2, dropout=0.1)

# Train model
train_model(model, optimizer, criterion, train_loader, val_loader, 
           model_dict, param_file, epochs=100)
```

### Model Deployment
```python
from torchserve import generate_jit_model, pack_model, register_model

# Generate TorchScript model
generate_jit_model('model_config.pkl', 'output_path')

# Pack for deployment
pack_model('model_dir', 'model.pt', version='1.0', model_store='model_store')

# Register with TorchServe
register_model('model_name', '1.0')
```

## Applications

- **Mesh Quality Assessment**: Evaluate computational mesh quality for CFD simulations
- **Mesh Optimization**: Identify problematic mesh regions for refinement
- **Automated Mesh Generation**: Quality control in mesh generation pipelines
- **Scientific Computing**: Integration with computational fluid dynamics workflows

## Dependencies

- Python 3.8+
- PyTorch 1.12.1+ (with CUDA support recommended)
- PyTorch Geometric 2.0.4+
- NumPy, SciPy, Pandas for data processing
- TorchServe for model deployment

## License

This project is part of the MEGAS2023 research work. Please refer to the original paper for detailed methodology and experimental results.

