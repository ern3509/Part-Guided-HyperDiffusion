import os
import json
import numpy as np
import trimesh
from tqdm import tqdm
import open3d as o3d
from glob import glob

def merge_parts_with_labels(obj_dir):
    meshes = []
    labels = []
    part_names = sorted([f for f in os.listdir(obj_dir) if f.endswith(".obj")])
    for label, file in enumerate(part_names):
        mesh = trimesh.load(os.path.join(obj_dir, file), force='mesh')
        if not mesh.is_empty:
            meshes.append(mesh)
            labels.append(np.full(len(mesh.faces), label))  # assign label per face
    if not meshes:
        raise RuntimeError(f"No mesh parts found in {obj_dir}")
    combined_mesh = trimesh.util.concatenate(meshes)
    face_labels = np.concatenate(labels)
    return combined_mesh, face_labels

def compute_sdf(points, mesh, batch_size=100000):
    from trimesh.proximity import signed_distance
    sdf_vals = []
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        dists = signed_distance(mesh, batch)
        sdf_vals.append(dists)
    return np.concatenate(sdf_vals, axis=0)

def simplify_trimesh(tri_mesh, target_faces=100000):
    if len(tri_mesh.faces) <= target_faces:
        return tri_mesh  # No need to simplify

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    mesh_o3d.compute_vertex_normals()

    simplified = mesh_o3d.simplify_quadric_decimation(target_faces)
    simplified.remove_unreferenced_vertices()

    simplified_trimesh = trimesh.Trimesh(
        vertices=np.asarray(simplified.vertices),
        faces=np.asarray(simplified.triangles)
    )
    return simplified_trimesh

def sample_points_per_part(obj_dir, num_points_per_part=1000, noise_std=0.01):
    point_list = []
    label_list = []

    part_files = sorted([f for f in os.listdir(obj_dir) if f.endswith(".obj")])
    for label, file in enumerate(part_files):
        mesh = trimesh.load(os.path.join(obj_dir, file), force='mesh')
        if mesh.is_empty:
            continue

        num_points = min(num_points_per_part, mesh.faces.shape[0] * 10)  # safeguard
        points = mesh.sample(num_points)
        noise = np.random.normal(scale=noise_std, size=points.shape)
        points += noise

        point_list.append(points)
        label_list.append(np.full((points.shape[0],), label))

    if not point_list:
        raise RuntimeError(f"No valid meshes found in {obj_dir}")

    pc_surface = np.vstack(point_list)
    part_labels = np.concatenate(label_list)

    return pc_surface, part_labels

def process_shape(root_dir, shape_id, output_dir, num_surface=5000, num_random=5000, noise_std=0.01):
    obj_dir = os.path.join(root_dir, shape_id, "objs")
    nb_parts = len(os.listdir(obj_dir))
    if not os.path.exists(obj_dir):
        print(f"❌ No 'objs/' folder in {shape_id}, skipping.")
        return 0

    try:
        pc_surface, part_labels_surface = sample_points_per_part(obj_dir, num_surface // nb_parts, noise_std=noise_std)
        merged_mesh = trimesh.util.concatenate(
            [trimesh.load(os.path.join(obj_dir, f), force='mesh') for f in os.listdir(obj_dir) if f.endswith(".obj")]
        )
        merged_mesh = simplify_trimesh(merged_mesh, target_faces=50000)
    except Exception as e:
        print(f"❌ Failed processing mesh for {shape_id}: {e}")
        return 0

    # Random space samples
    pc_random = np.random.uniform(low=-1, high=1, size=(num_random, 3))
    part_labels_random = np.full((num_random,), -1)

    # Combine and compute SDF
    points = np.vstack([pc_surface, pc_random])
    part_labels = np.concatenate([part_labels_surface, part_labels_random])
    sdf = compute_sdf(points, merged_mesh)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{shape_id}.npy")
    np.save(output_path, np.hstack([points, sdf.reshape(-1, 1), part_labels.reshape(-1, 1)]))
    print(f"✅ Saved: {output_path}")
    return 1


def load_train_split(split_path):
    with open(split_path, "r") as f:
        split = json.load(f)
    return [entry["anno_id"] for entry in split]

if __name__ == "__main__":
    split_path = "./data/knife/Knife.train.json"
    partnet_dir = "./data/knife"
    output_dir = "./data/knife_preprocessed"

    model_ids = load_train_split(split_path)
    print("Starting...")
    i = 0
    existing_ids = [f[:-4] for f in os.listdir(output_dir) if f.endswith(".npy")]

    for shape_id in tqdm(model_ids, desc="Processing shapes"):
        if shape_id in existing_ids:
            continue
        print(shape_id)
        l = process_shape(partnet_dir, shape_id, output_dir)
        i += l
    print(f"Done, there are {i} files also in train knives")
