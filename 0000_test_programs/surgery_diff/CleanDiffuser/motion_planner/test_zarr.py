import zarr
import numpy as np

root = zarr.open('0000_test_programs/surgery_diff/CleanDiffuser/datasets/test.zarr', mode='r')
print('done')