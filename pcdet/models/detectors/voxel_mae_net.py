from .detector3d_template_voxel_mae import Detector3DTemplate_voxel_mae
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
import numpy as np


class Voxel_MAE(Detector3DTemplate_voxel_mae):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            # Only perform post-processing if not counting FLOPs
            if not hasattr(self, 'counting_flops'):
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts
            else:
                return batch_dict
            
    def get_training_loss(self):
        disp_dict = {} 

        loss_rpn, tb_dict = self.backbone_3d.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
    
    def count_flops_manually(self, batch_dict):
        total_flops = 0
        
        # Count FLOPs for each module
        for module in self.module_list:
            if hasattr(module, 'count_flops'):
                total_flops += module.count_flops(batch_dict)
            elif isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                # Manual calculation for Conv2d and ConvTranspose2d
                out_h, out_w = module.output_size
                total_flops += out_h * out_w * module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1] / module.groups
        
        print(f"Estimated total FLOPs: {total_flops}")
        return total_flops
    

    def count_parameters_and_flops(self, sample_batch_dict):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        def count_conv2d(m, x, y):
            x = x[0]  # remove tuple
            cin = m.in_channels
            cout = m.out_channels
            kh, kw = m.kernel_size
            batch_size = x.size()[0]
            out_h = y.size(2)
            out_w = y.size(3)
            return batch_size * out_h * out_w * cin * cout * kh * kw

        def count_bn2d(m, x, y):
            x = x[0]  # remove tuple
            return 2 * x.numel()

        def count_relu(m, x, y):
            x = x[0]  # remove tuple
            return x.numel()

        def count_softmax(m, x, y):
            x = x[0]  # remove tuple
            return x.numel()

        def count_maxpool(m, x, y):
            x = x[0]  # remove tuple
            return 0

        def count_avgpool(m, x, y):
            x = x[0]  # remove tuple
            return 0

        def count_linear(m, x, y):
            x = x[0]  # remove tuple
            return m.in_features * m.out_features

        registry = {
            'Conv2d': count_conv2d,
            'BatchNorm2d': count_bn2d,
            'ReLU': count_relu,
            'ReLU6': count_relu,
            'Softmax': count_softmax,
            'MaxPool2d': count_maxpool,
            'AvgPool2d': count_avgpool,
            'Linear': count_linear
        }

        def count_flops(module, input, output):
            module_type = type(module).__name__
            if module_type in registry:
                return registry[module_type](module, input, output)
            else:
                return 0

        total_flops = 0
        for name, module in self.named_modules():
            module.register_forward_hook(lambda m, i, o: setattr(m, 'total_flops', count_flops(m, i, o)))
        
        # Run a forward pass with the sample batch dictionary
        with torch.no_grad():
            self(sample_batch_dict)

        # Sum up the flops for each module
        for name, module in self.named_modules():
            if hasattr(module, 'total_flops'):
                total_flops += module.total_flops

        return total_params, total_flops


    # def count_flops(self, batch_dict):
    #     self.eval()
    #     device = next(self.parameters()).device
    #     print(f"Model is on device: {device}")
        
    #     # Process batch_dict (move tensors to the correct device)
    #     processed_batch_dict = {}
    #     for k, v in batch_dict.items():
    #         if isinstance(v, np.ndarray):
    #             if v.dtype.kind in ['U', 'S']:  # String arrays
    #                 processed_batch_dict[k] = v  # Keep string arrays as-is
    #             else:
    #                 processed_batch_dict[k] = torch.from_numpy(v).to(device)
    #         elif isinstance(v, torch.Tensor):
    #             processed_batch_dict[k] = v.to(device)
    #         else:
    #             processed_batch_dict[k] = v

    #     # Create a wrapper that only runs the forward pass without post-processing
    #     class WrapperModule(nn.Module):
    #         def __init__(self, model):
    #             super().__init__()
    #             self.model = model

    #         def forward(self, batch_dict):
    #             for cur_module in self.model.module_list:
    #                 batch_dict = cur_module(batch_dict)
    #             return batch_dict

    #     wrapper_model = WrapperModule(self)

    #     # Use PyTorch Profiler with CUDA events
    #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
    #                 record_shapes=True,
    #                 with_flops=True,
    #                 profile_memory=True) as prof:
    #         with record_function("model_inference"):
    #             _ = wrapper_model(processed_batch_dict)

    #     # Print profiler results
    #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
    #     # Export the results to a file
    #     prof.export_chrome_trace("trace.json")

    #     # Manual FLOP counting
    #     manual_flops = self.count_flops_manually(processed_batch_dict)
    #     print(f"Manually estimated FLOPs: {manual_flops}")
        
    #     return prof, manual_flops





    def count_flops(self, batch_dict):
        self.eval()
        device = next(self.parameters()).device
        print(f"Model is on device: {device}")
        
        # Move batch_dict to the correct device and convert numpy arrays to tensors
        sample_batch_dict = {}
        for k, v in batch_dict.items():
            if isinstance(v, np.ndarray):
                if v.dtype.kind in ['U', 'S']:  # String arrays
                    sample_batch_dict[k] = v  # Keep string arrays as-is
                else:
                    sample_batch_dict[k] = torch.from_numpy(v).to(device)
            elif isinstance(v, torch.Tensor):
                sample_batch_dict[k] = v.to(device)
            else:
                sample_batch_dict[k] = v

        # Print shapes for debugging
        for k, v in sample_batch_dict.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                print(f"{k} shape: {v.shape}")
            else:
                print(f"{k} type: {type(v)}")

        # Set the counting_flops attribute
        self.counting_flops = True

        total_params, total_flops = self.count_parameters_and_flops(sample_batch_dict)
        
        # Use PyTorch Profiler for timing information
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True) as prof:
            with record_function("model_inference"):
                _ = self(sample_batch_dict)

        # Unset the counting_flops attribute
        delattr(self, 'counting_flops')

        # Print profiler results
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        
        # Export the results to a file
        prof.export_chrome_trace("trace.json")

        return prof, total_params, total_flops





    # def count_flops(self, batch_dict):
    #     self.eval()
    #     device = next(self.parameters()).device
    #     print(f"Model is on device: {device}")
        
    #     # Process batch_dict (move tensors to the correct device)
    #     processed_batch_dict = {}
    #     for k, v in batch_dict.items():
    #         if isinstance(v, np.ndarray):
    #             if v.dtype.kind in ['U', 'S']:  # String arrays
    #                 processed_batch_dict[k] = v  # Keep string arrays as-is
    #             else:
    #                 processed_batch_dict[k] = torch.from_numpy(v).to(device)
    #         elif isinstance(v, torch.Tensor):
    #             processed_batch_dict[k] = v.to(device)
    #         else:
    #             processed_batch_dict[k] = v

    #     # Create a wrapper that only runs the forward pass without post-processing
    #     class WrapperModule(nn.Module):
    #         def __init__(self, model):
    #             super().__init__()
    #             self.model = model
    #             self.model.counting_flops = True  # Set flag to bypass post-processing

    #         def forward(self, batch_dict):
    #             return self.model(batch_dict)

    #         def __del__(self):
    #             if hasattr(self.model, 'counting_flops'):
    #                 del self.model.counting_flops

    #     wrapper_model = WrapperModule(self)

    #     # Use PyTorch Profiler
    #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #                 record_shapes=True,
    #                 profile_memory=True,
    #                 with_flops=True) as prof:
    #         with record_function("model_inference"):
    #             _ = wrapper_model(processed_batch_dict)

    #     # Print profiler results
    #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        
    #     # Export the results to a file
    #     prof.export_chrome_trace("trace.json")

    #     # Estimate FLOPs
    #     total_flops = 0
    #     for event in prof.key_averages():
    #         if event.flops is not None:
    #             total_flops += event.flops

    #     return prof, total_flops