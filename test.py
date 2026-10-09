'''
Author: xiafanli xiafanli@eccom.com.cn
Date: 2023-10-24 11:53:58
Description: 
'''
import torch
import torch.nn as nn
import numpy as np
import h5py
from h5py import Dataset, Group, File

with h5py.File(r"F:\datasets\water_datasets\LSPIV-measurements\Dataset 2\Config_BD.h5","r") as f:
    def print_attrs(name, obj):
        print(name)
        for key, val in obj.attrs.items():
            print("    %s: %s" % (key, val))

    f.visititems(print_attrs)

