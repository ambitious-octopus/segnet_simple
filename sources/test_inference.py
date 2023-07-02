#!/usr/bin/env python3

import torch
import ros_inference

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seg_class_names = [
                'power_drill',
                'screwdriver',
                'plate',
    ]

    seg_model = ros_inference.PrednetSegmentation(name='n', gpu_factor=0.1, device=device, class_names=seg_class_names, with_ros=False)
    seg_model.TestFile(filein='test_img.png', fileout_base='seg_img')
