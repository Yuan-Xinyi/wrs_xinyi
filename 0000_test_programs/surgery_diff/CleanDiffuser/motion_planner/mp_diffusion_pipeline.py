import zarr

root = zarr.open('0000_test_programs/surgery_diff/CleanDiffuser/datasets/xarm_toppra_mp.zarr')
print(root.tree())