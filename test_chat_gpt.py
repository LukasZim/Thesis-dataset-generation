
import numpy as np

# Example arrays
original_array = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])
target_array = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90],
    [90, 110, 130]
])

# Compute min and max for each row of the original array
orig_min = np.min(original_array, axis=0)
print(orig_min)
orig_max = np.max(original_array, axis=0)
print(orig_max)
# Compute min and max for each row of the target array
target_min = np.min(target_array, axis=0)
target_max = np.max(target_array, axis=0)

# Rescale each column of the original array based on the target min and max
rescaled_array = (original_array - orig_min) / (orig_max - orig_min) * (target_max - target_min) + target_min

print("Original Array:")
print(original_array)
print("Target Array:")
print(target_array)
print("Rescaled Array:")
print(rescaled_array)