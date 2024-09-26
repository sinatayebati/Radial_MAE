from functools import partial
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils
from .spconv_backbone import post_act_block


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


class Radial_MAE(nn.Module):
    """
    pre-trained model with radial masking
    """

    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1]+ [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        self.masked_ratio = model_cfg.MASKED_RATIO
        self.angular_range = model_cfg.get('ANGULAR_RANGE', 5)  # Default to 5 if not specified

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
            last_pad = self.model_cfg.get('last_pad', 0)

            self.conv_out = spconv.SparseSequential(
                # [200, 150, 5] -> [200, 150, 2]
                spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                    bias=False, indice_key='spconv_down2'),
                norm_fn(128),
                nn.ReLU(),
            )
        else:
            self.conv_out = None

        self.num_point_features = 16
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(128, 32, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(32, 8, 3, padding=1, output_padding=1, stride=(4,2,2), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(8, 1, 3, padding=1, output_padding=1, stride=(3,2,2), bias=False),
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.forward_re_dict = {}

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        pred = self.forward_re_dict['pred']
        target = self.forward_re_dict['target']
        loss = self.criterion(pred, target)

        tb_dict = {
            'loss_rpn': loss.item()
        }

        return loss, tb_dict


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """


        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']

        select_ratio = 1 - self.masked_ratio # ratio for select voxel
        
        # Calculate angles in radians for each voxel
        angles = torch.atan2(voxel_coords[:, 3], voxel_coords[:, 2])  # atan2(y, x)
        angles_deg = torch.rad2deg(angles) % 360  # Convert to degrees and ensure in range [0, 360)

        # Group indices based on angles (adjustable angular range)
        angular_range = self.angular_range  # degrees, can be adjusted
        radial_groups = {}
        for angle in range(0, 360, angular_range):
            mask = (angles_deg >= angle) & (angles_deg < angle + angular_range)
            group_indices = torch.where(mask)[0]
            if len(group_indices) > 0:  # Only consider non-empty groups
                radial_groups[angle] = group_indices
        
        # Ensure there are enough groups to select from
        if len(radial_groups) == 0:
            raise ValueError("No non-empty radial groups found. Consider adjusting the selection criteria.")
        
        # Randomly select a portion of radial groups
        select_ratio = 1 - self.masked_ratio  # Assuming masked_ratio is defined
        num_groups_to_select = min(int(select_ratio * len(radial_groups)), len(radial_groups))
        selected_group_angles = random.sample(list(radial_groups.keys()), num_groups_to_select)

        # Combine indices from the selected groups
        selected_indices = []
        for angle in selected_group_angles:
            selected_indices.extend(radial_groups[angle].tolist())

        # Ensure a minimum number of voxels are selected
        min_voxels = max(1, int(0.1 * voxel_features.shape[0]))  # Ensuring at least 10% or 1 voxel is selected
        if len(selected_indices) < min_voxels:
            selected_indices = random.sample(range(voxel_features.shape[0]), min_voxels)
        
        # Convert list to 1-dimensional tensor
        selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=voxel_coords.device)


        # nums is set to voxel_features.shape[0], which is the number of voxels (the size of the first dimension of voxel_features)
        nums = voxel_features.shape[0]
        
        """
        torch.ones(nums, 1) creates a 2D tensor of shape [nums, 1] where every element is 1
        The purpose of this tensor seems to be to create a column vector of ones with the same number
        of rows as there are voxels. This could be used for operations that require a tensor of ones,
        possibly for masking or normalization purposes.
        """
        voxel_fratures_all_one = torch.ones(nums,1).to(voxel_features.device)


        """
        1- voxel_features_partial:
            This tensor is a subset of voxel_features indexed by slect_index
            voxel_features is initially provided in batch_dict with shape [num_voxels, C], where num_voxels
            is the total number of voxels and C is the number of channels or features per voxel.
            slect_index is a 1D tensor containing selected indices based on certain criteria related to voxel
            coordinates distance. It is formed by concatenating indices from three different conditions (select_30, select_30to50, select_50) and then applying a selection ratio.
            voxel_features_partial will have a shape [len(slect_index), C]. It represents a subset of the voxel features based on the selected indices.

        2- voxel_coords_partial:
            Similar to voxel_features_partial, this tensor is a subset of voxel_coords indexed by slect_index.
            voxel_coords is provided in batch_dict with shape [num_voxels, 4], representing the coordinates of each voxel. The coordinates are structured as [batch_idx, z_idx, y_idx, x_idx].
            voxel_coords_partial will have a shape [len(slect_index), 4]. It represents the coordinates of the selected voxels.
        """
        # Use the selected indices to segment voxel_features and voxel_coords
        voxel_features_partial = voxel_features[selected_indices_tensor, :] # shape [N, C]
        voxel_coords_partial = voxel_coords[selected_indices_tensor, :] # shape [N, 4]


        # batch_size = batch_dict['batch_size']
        # TEMP CODE
        batch_size = batch_dict.get('batch_size', 4)
        """
        spconv.SparseConvTensor:
            Args:
                features = # your features with shape [N, num_channels]
                indices = # your indices/coordinates with shape [N, ndim + 1], batch index must be put in indices[:, 0]
                spatial_shape = # spatial shape of your sparse tensor, spatial_shape[i] is shape of indices[:, 1 + i].
                batch_size = # batch size of your sparse tensor.
                x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
                x_dense_NCHW = x.dense() # convert sparse tensor to dense NCHW tensor.
        """
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features_partial,
            indices=voxel_coords_partial.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        input_sp_tensor_ones = spconv.SparseConvTensor(
            features=voxel_fratures_all_one,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        # print(f"Shape of out before dense(): {out.shape}")
        dense_out = out.dense()
        print(f"Shape of dense_out: {dense_out.shape}")
        # Shape of dense_out: torch.Size([4, 128, 2, 200, 176])

        self.forward_re_dict['target'] = input_sp_tensor_ones.dense()
        x_up1 = self.deconv1(out.dense())
        print(f"Shape of x_up1: {x_up1.shape}")
        # Shape of x_up1: torch.Size([4, 32, 4, 400, 352])
        x_up2 = self.deconv2(x_up1)
        x_up3 = self.deconv3(x_up2)
   
        self.forward_re_dict['pred'] = x_up3

        return batch_dict