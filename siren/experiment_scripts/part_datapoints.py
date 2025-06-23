import os
import numpy as np
import trimesh
import open3d as o3d
import igl
from tqdm import tqdm
import json


def normalize_and_center_mesh(mesh):
    vertices = mesh.vertices.copy()
    vertices -= np.mean(vertices, axis=0, keepdims=True)
    scale = 0.5 * 0.95 / np.max(np.abs(vertices))
    vertices *= scale
    mesh.vertices = vertices
    return mesh

def simplify_trimesh(tri_mesh, target_faces=50000):
    if len(tri_mesh.faces) <= target_faces:
        return tri_mesh
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    mesh_o3d.compute_vertex_normals()
    simplified = mesh_o3d.simplify_quadric_decimation(target_faces)
    simplified.remove_unreferenced_vertices()
    return trimesh.Trimesh(vertices=np.asarray(simplified.vertices), faces=np.asarray(simplified.triangles))

def sample_global_points(mesh, total=30000, noise_std=0.01):
    n_surface = total // 3
    n_near = total // 3
    n_far = total - n_surface - n_near
    surface = mesh.sample(n_surface)
    near_surface = surface + noise_std * np.random.randn(*surface.shape)
    far = np.random.uniform(-0.5, 0.5, size=(n_far, 3))
    return np.vstack([surface, near_surface, far])

def load_part_meshes(obj_dir):
    meshes = []
    for file in sorted(os.listdir(obj_dir)):
        if file.endswith(".obj"):
            mesh = trimesh.load(os.path.join(obj_dir, file), force='mesh')
            if not mesh.is_empty:
                meshes.append(mesh)
    return meshes

def assign_part_labels(points, part_meshes, occupancy):
    labels = np.full(len(points), -1)
    for pid, mesh in enumerate(part_meshes):
        wn = igl.winding_number(mesh.vertices, mesh.faces, points)
        occupancy_mask = occupancy == 1  # convert float array to boolean
        mask = (wn >= 0.5) & (labels == -1) & occupancy_mask
        labels[mask] = pid
        print(f"Assigned {mask.sum()} points to part {pid}")
        print(f"Part {pid} volume: {mesh.volume:.4f}")
    return labels

def process_shape(root_dir, shape_id, output_dir, num_points=100000, noise_std=0.01):
    obj_dir = os.path.join(root_dir, shape_id, "objs")
    if not os.path.exists(obj_dir):
        print(f"❌ Missing 'objs/' folder for {shape_id}, skipping.")
        return 0

    print(f"Processing {shape_id}...")
    try:
        part_meshes = load_part_meshes(obj_dir)
        if not part_meshes:
            raise RuntimeError("No valid part meshes.")

        merged_mesh = trimesh.util.concatenate(part_meshes)
        merged_mesh = simplify_trimesh(merged_mesh, target_faces=50000)
        merged_mesh = normalize_and_center_mesh(merged_mesh)

        # Sample points so that each part is rightly represented
        points = sample_global_points(merged_mesh, total=num_points, noise_std=noise_std)
        occupancy = (igl.winding_number(merged_mesh.vertices, merged_mesh.faces, points) >= 0.5).astype(np.float32)
        print(f" Inside: {(occupancy == 1).sum()}, Outside: {(occupancy == 0).sum()}")
        labels = assign_part_labels(points, part_meshes, occupancy)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{shape_id}.npy")
        np.save(output_path, np.hstack([points, occupancy[:, None], labels[:, None]]))

        print(f"✅ Saved {output_path} | Inside: {(occupancy == 1).sum()}, Outside: {(occupancy == 0).sum()}")
        print(f"Labeled inside: {(labels != -1).sum()}, total inside: {occupancy.sum()}")

        return 1

    except Exception as e:
        print(f"❌ Failed on {shape_id}: {e}")
        return 0

def load_train_split(split_path):
    with open(split_path, "r") as f:
        split = json.load(f)
    return [entry["anno_id"] for entry in split]

if __name__ == "__main__":
    split_path = "./data/knife/Knife.train.json"
    partnet_dir = "./data/knife"
    output_dir = "./data/knife_preprocessed_split"

    model_ids = load_train_split(split_path)
    existing_ids = {f[:-4] for f in os.listdir(output_dir) if f.endswith(".npy")}
    print("Starting preprocessing...")

    count = 0
    for shape_id in tqdm(model_ids, desc="Shapes"):
        if shape_id in existing_ids:
            continue
        count += process_shape(partnet_dir, shape_id, output_dir)

    print(f"Done. Successfully processed {count} shapes.")
