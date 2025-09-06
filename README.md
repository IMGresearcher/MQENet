# MQENet

A Mesh Quality Evaluation Neural Network Based on Dynamic Graph Attention

## File Structure

### Core Files

- **`model.py`** - Contains the main neural network architecture
  - `DeepGCNLayer`: Custom graph convolutional layer with residual connections
  - `DeeperGCN`: Main model implementing a deep graph convolutional network
  - `TraceModel`: Wrapper for model tracing and deployment

- **`preprocess.py`** - Mesh preprocessing and graph conversion utilities
  - `grd2vertex_graph()`: Converts 2D GRD meshes to vertex-based graphs
  - `grd2element_graph()`: Converts 2D GRD meshes to element-based graphs
  - `compute_proximity_adjacency()`: Creates adjacency matrices based on spatial proximity
  - `preprocess_grd_meshes()`: Batch processing for GRD files

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

### Training Models
```python
from mesh_dataset import MeshDataset
from model import DeeperGCN
from utils import train_model

# Create dataset
dataset = MeshDataset(root='data', raw_file_root='processed', 
                          num=1024, label_func=grd_label)

# Initialize model
model = DeeperGCN(input_channels=6, num_layers=4, hidden_channels=12, 
                  num_classes=8, dropout=0.01)

from utils import train_model,add_weight_decay,\
    split_dataset,test_model,print_results,save_model

train_data , val_data ,test_data = split_dataset(mesh)

add_weight_decay(model=model,weight_decay=0.0001)

train_model(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=0.01,amsgrad=True),
            criterion=nn.CrossEntropyLoss(),train_loader=DataLoader(train_data,batch_size=32, shuffle=True),
            val_loader=DataLoader(val_data,batch_size=32, shuffle=True),epochs=5000,patience=25,writer=writer,
            lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(torch.optim.Adam(model.parameters()), mode='max',
                                                                    factor=0.1,
                                                                    patience=10, verbose=True, threshold=0.0001,
                                                                    threshold_mode='rel', cooldown=0, min_lr=1e-09,
                                                                    eps=1e-09),
            grad_clipping_value=torch.nn.utils.clip_grad_norm(model.parameters(), 20, norm_type=2))

```

## Dependencies
- Python 3.8+
- PyTorch 1.12.1+ (with CUDA support recommended)
- PyTorch Geometric 2.0.4+









