import zarr
import numpy as np

root = zarr.open('0000_test_programs/surgery_diff/CleanDiffuser/datasets/franka_kinodyn_obstacles_3.zarr', mode='r')
print('done')