import zarr
import numpy as np

root = zarr.open('0000_test_programs/surgery_diff/CleanDiffuser/datasets/franka_mp_ruckig_1000hz.zarr', mode='r')
print('done')