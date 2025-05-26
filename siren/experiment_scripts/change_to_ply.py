import os
import json
import numpy as np
import trimesh
from tqdm import tqdm
import open3d as o3d
from glob import glob
import matplotlib.pyplot as plt

data = np.load("./data/knife_preprocessed/944.npy")
points = data[:, :3]
sdf = data[:, 3]

# Normalize SDF to [0, 1] and convert to color
sdf_normalized = (sdf - sdf.min()) / (sdf.max() - sdf.min())
colors = (plt.cm.jet(sdf_normalized)[:, :3] * 255).astype(np.uint8)  # RGB colormap

# Write to colored PLY
with open("your_shape_id_colored.ply", "w") as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {points.shape[0]}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
    for p, c in zip(points, colors):
        f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
