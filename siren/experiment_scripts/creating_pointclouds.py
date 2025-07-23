import os
import json
import numpy as np
import trimesh
from tqdm import tqdm
import open3d as o3d
from glob import glob
import igl


def merge_parts_of_multiple_objects(obj_dir):
    for file in os.listdir(obj_dir):
        print(f"Processing {file}...")
        obj_file = os.path.join(obj_dir, file)
        if os.path.isdir(obj_file):
            objs_path = os.path.join(obj_file, "objs")
            if os.path.isdir(objs_path):
                full_obj = merge_parts(objs_path)
        if full_obj.is_empty:
            raise RuntimeError(f"Empty mesh found in {obj_file}")
        full_obj.export(os.path.join(obj_dir, f"{file}.obj"))
                
def merge_parts(obj_dir):
    meshes = []
    for file in os.listdir(obj_dir):
        if file.endswith(".obj"):
            mesh = trimesh.load(os.path.join(obj_dir, file), force='mesh')
            if not mesh.is_empty:
                meshes.append(mesh)
    if not meshes:
        raise RuntimeError(f"No mesh parts found in {obj_dir}")
    return trimesh.util.concatenate(meshes)


def compute_sdf(points, mesh, batch_size=100000):
    from trimesh.proximity import signed_distance
    sdf_vals = []
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        dists = signed_distance(mesh, batch)
        sdf_vals.append(dists)
    return np.concatenate(sdf_vals, axis=0)

def simplify_trimesh(tri_mesh, target_faces=2500):
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

def normalize_and_center_mesh(mesh):
    vertices = mesh.vertices.copy()
    vertices -= np.mean(vertices, axis=0, keepdims=True)
    v_max = np.max(vertices)
    v_min = np.min(vertices)
    vertices *= 0.5 * 0.95 / max(abs(v_min), abs(v_max))
    mesh.vertices = vertices
    return mesh


def sample_and_label_winding(mesh, n_total=20000):
    mesh = normalize_and_center_mesh(mesh)
    n_surface = n_total // 3
    n_near = n_surface
    n_far = n_total - n_surface - n_near

    surface = mesh.sample(n_surface)
    near_surface = surface + 0.01 * np.random.randn(*surface.shape)
    far = np.random.uniform(-0.5, 0.5, size=(n_far, 3))

    all_points = np.vstack([surface, near_surface, far])
    wn_values = igl.winding_number(np.array(mesh.vertices), np.array(mesh.faces), all_points)
    occ = (wn_values >= 0.5).astype(np.float32)

    # Debug info
    print("Winding stats:")
    print("→ Inside:", (occ == 1).sum(), f"({100 * occ.mean():.2f}%) | Outside:", (occ == 0).sum(), f"({100 * (1 - occ.mean()):.2f}%)")

    return all_points, occ.reshape(-1, 1)


def process_shape(root_dir, shape_id, output_dir, output_type, num_total=20000, noise_std=0.01):
    obj_dir = os.path.join(root_dir, shape_id, "objs")
    if not os.path.exists(obj_dir):
        print(f"❌ No 'objs/' folder in {shape_id}, skipping.")
        return 0

    try:
        mesh = merge_parts(obj_dir)
        mesh = simplify_trimesh(mesh, target_faces=50000)
        mesh = normalize_and_center_mesh(mesh)
    except Exception as e:
        print(f"❌ Failed to merge mesh for {shape_id}: {e}")
        return 0

    try:
        points, occ = sample_and_label_winding(mesh, n_total=num_total)
        data = np.concatenate([points, occ], axis=1)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{shape_id}.npy")
        np.save(output_path, data)
        print(f"✅ Saved: {output_path}")
        return 1
    except Exception as e:
        print(f"❌ Winding number failed for {shape_id}: {e}")
        return 0


def load_train_split(split_path):
    with open(split_path, "r") as f:
        split = json.load(f)
    return [entry["anno_id"] for entry in split]

if __name__ == "__main__":
    # Customize these paths:
    split_path = "./data/knife/Knife.train.json"
    partnet_dir = "./data/knife"
    output_dir = "./data/knife_preprocessed"

    output_type = "occ"
    model_ids = load_train_split(split_path)
    print("Starting...")
    i = 0
    existing_ids = [f[:-4] for f in os.listdir(output_dir) if f.endswith(".npy")]
    existing_ids.extend(["1193", "1167"])
    print(existing_ids)
    for shape_id in tqdm(model_ids, desc="Processing shapes"):
        if shape_id in existing_ids:
            continue
        print(shape_id)
        l = process_shape(partnet_dir, shape_id, output_dir, output_type)
        i += l
    print(f"Done there are {i} files also in train knives")
