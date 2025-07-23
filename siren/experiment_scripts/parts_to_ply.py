import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def label_to_color(labels):
    """Assign a unique RGB color to each part label"""
    unique_labels = np.unique(labels)
    color_map = plt.cm.get_cmap("tab20", len(unique_labels))  # good for categorical
    label_to_rgb = {
        label: (np.array(color_map(i)[:3]) * 255).astype(np.uint8)
        for i, label in enumerate(unique_labels)
    }
    return np.array([label_to_rgb[label] for label in labels])

def convert_npy_to_colored_ply(npy_path, ply_path):
    data = np.load(npy_path)
    points = data[:, :3]
    labels = data[:, 4].astype(int)

    # Assign RGB color per part label
    colors = label_to_color(labels)

    with open(ply_path, "w") as f:
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

    print(f"✅ Saved: {ply_path}")

if __name__ == "__main__":
    input_dir = "./data/knife_preprocessed_split"
    output_dir = "./data/knife_colored_ply"
    os.makedirs(output_dir, exist_ok=True)

    for npy_file in glob(os.path.join(input_dir, "*.npy")):
        shape_id = os.path.splitext(os.path.basename(npy_file))[0]
        ply_path = os.path.join(output_dir, f"{shape_id}.ply")
        convert_npy_to_colored_ply(npy_file, ply_path)
