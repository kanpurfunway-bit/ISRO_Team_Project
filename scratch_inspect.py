import numpy as np
import os
try:
    data = np.load('Data/normalized_with_season-001.npy', allow_pickle=True)
    print("Shape:", data.shape)
    print("Type:", data.dtype)
    print("First item:", data[0])
except Exception as e:
    print("Error:", e)
