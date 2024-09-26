import sys
import os
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
from functools import partial
import torch.nn as nn

# Import your model (adjust the import path as needed)
from pcdet.models.backbones_3d.radial_mae import Radial_MAE

def count_flops():
    # Create a dummy config for the model
    class DummyConfig(dict):
        def __getattr__(self, name):
            return self.get(name, None)

    model_cfg = DummyConfig({
        'MASKED_RATIO': 0.5,
        'ANGULAR_RANGE': 5,
        'RETURN_ENCODED_TENSOR': True
    })

    # Define grid_size to match your model's expectations
    grid_size = [1408, 1600, 40]  # Adjust these values based on your actual grid size

    # Create an instance of your model
    model = Radial_MAE(
        model_cfg=model_cfg,
        input_channels=4,  # Adjust this based on your input
        grid_size=grid_size,
        voxel_size=[0.05, 0.05, 0.1],  # Adjust these values as needed
        point_cloud_range=[0, -40, -3, 70.4, 40, 1]  # Adjust these values as needed
    )
    model.eval()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move the model to the appropriate device
    model = model.to(device)

    # Create a dummy batch_dict
    batch_size = 1
    num_voxels = 100  # Adjust this based on your typical input
    voxel_features = torch.randn(num_voxels, 4, device=device)  # Assuming 4 input channels
    voxel_coords = torch.randint(0, 100, (num_voxels, 4), device=device)  # [batch_idx, z_idx, y_idx, x_idx]

    batch_dict = {
        'batch_size': batch_size,
        'voxel_features': voxel_features,
        'voxel_coords': voxel_coords
    }

    # Use FlopCountAnalysis
    flop = FlopCountAnalysis(model, batch_dict)
    
    print(flop_count_table(flop, max_depth=4))
    print(flop_count_str(flop))
    print(f"Total FLOPs: {flop.total()}")

if __name__ == '__main__':
    count_flops()