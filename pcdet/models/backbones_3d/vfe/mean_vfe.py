# import torch

# from .vfe_template import VFETemplate


# class MeanVFE(VFETemplate):
#     def __init__(self, model_cfg, num_point_features, **kwargs):
#         super().__init__(model_cfg=model_cfg)
#         self.num_point_features = num_point_features

#     def get_output_feature_dim(self):
#         return self.num_point_features

#     def forward(self, batch_dict, **kwargs):
#         """
#         Args:
#             batch_dict:
#                 voxels: (num_voxels, max_points_per_voxel, C)
#                 voxel_num_points: optional (num_voxels)
#             **kwargs:

#         Returns:
#             vfe_features: (num_voxels, C)
#         """
#         voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
#         print(f"voxel_features shape: {voxel_features.shape}")
#         print(f"voxel_num_points shape: {voxel_num_points.shape}")  
#         points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
#         normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
#         points_mean = points_mean / normalizer
#         batch_dict['voxel_features'] = points_mean.contiguous()

#         return batch_dict


import torch
import numpy as np

from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    # def forward(self, batch_dict, **kwargs):
    #     voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
    #     print(f"voxel_features shape: {voxel_features.shape}")
    #     print(f"voxel_num_points shape: {voxel_num_points.shape}")

    #     if voxel_features.dim() == 2:
    #         # If voxel_features is already 2D, we assume it's already processed
    #         points_mean = voxel_features
    #     else:
    #         # Original processing for 3D voxel_features
    #         points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
    #         normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
    #         points_mean = points_mean / normalizer

    #     batch_dict['voxel_features'] = points_mean.contiguous()

    #     return batch_dict

    def forward(self, batch_dict, **kwargs):
        voxel_features = batch_dict['voxels']
        voxel_num_points = batch_dict['voxel_num_points']
        
        # Convert to PyTorch tensor if numpy array
        if isinstance(voxel_features, np.ndarray):
            if voxel_features.dtype.kind not in ['U', 'S']:  # Not a string array
                voxel_features = torch.from_numpy(voxel_features).to(self.device)
        if isinstance(voxel_num_points, np.ndarray):
            if voxel_num_points.dtype.kind not in ['U', 'S']:  # Not a string array
                voxel_num_points = torch.from_numpy(voxel_num_points).to(self.device)

        if isinstance(voxel_features, torch.Tensor) and isinstance(voxel_num_points, torch.Tensor):
            points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
            points_mean = points_mean / normalizer
            batch_dict['voxel_features'] = points_mean.contiguous()
        else:
            print(f"Warning: voxel_features or voxel_num_points is not a tensor. Types: {type(voxel_features)}, {type(voxel_num_points)}")

        return batch_dict
