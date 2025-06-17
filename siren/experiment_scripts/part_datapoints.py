import os
import json
import numpy as np
import trimesh
from tqdm import tqdm
import open3d as o3d
from glob import glob
import igl

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

def simplify_trimesh(tri_mesh, target_faces=50000):
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

def sample_points_per_part(output_dir, obj_dir, total_points=30000, noise_std=0.01):
    point_list = []
    label_list = []
    occupancy_list = []
    part_files = sorted([f for f in os.listdir(obj_dir) if f.endswith(".obj")])
    n_parts = len(part_files)
    points_per_part = total_points // n_parts

    for label, file in enumerate(part_files):
        mesh = trimesh.load(os.path.join(obj_dir, file), force='mesh')
        if mesh.is_empty:
            continue
       # mesh = normalize_and_center_mesh(mesh)

        n_surface = points_per_part // 3
        n_near = n_surface
        n_far = points_per_part - n_surface - n_near

        surface = mesh.sample(n_surface)
        near_surface = surface + noise_std * np.random.randn(*surface.shape)
        far = np.random.uniform(-0.5, 0.5, size=(n_far, 3))
        all_points = np.vstack([surface, near_surface, far])

        wn = igl.winding_number(mesh.vertices, mesh.faces, all_points)
        occ = (wn >= 0.5).astype(np.float32)

        labeled_points = all_points[occ == 1]
        occ = occ[occ == 1]
        labels = np.full((labeled_points.shape[0],), label)
        point_list.append(labeled_points)
        label_list.append(labels)
        occupancy_list.append(occ)
        print(f"length of occ: {len(occupancy_list)} /n length of points: {len(point_list)}")
        #labels = np.full((labeled_points.shape[0],), label)
        #point_list.append(labeled_points)
        #label_list.append(labels)
        output_path = os.path.join(output_dir, f"{file}.npy")
        np.save(output_path, np.hstack([labeled_points, occ.reshape(-1, 1), labels.reshape(-1, 1)]))

    if not point_list:
        raise RuntimeError(f"No valid meshes in {obj_dir}")
    
    return np.vstack(point_list), np.concatenate(occupancy_list),np.concatenate(label_list)


def process_shape(root_dir, shape_id, output_dir, num_surface=5000, num_random=5000, noise_std=0.01):
    obj_dir = os.path.join(root_dir, shape_id, "objs")
    nb_parts = len(os.listdir(obj_dir))
    if not os.path.exists(obj_dir):
        print(f"❌ No 'objs/' folder in {shape_id}, skipping.")
        return 0
    print(f"Processing {shape_id} with {nb_parts} parts")
    try:
        pc, occupancies, part_labels_surface  = sample_points_per_part(output_dir, obj_dir, total_points=num_surface, noise_std=noise_std)

        merged_mesh = trimesh.util.concatenate(
            [trimesh.load(os.path.join(obj_dir, f), force='mesh') for f in os.listdir(obj_dir) if f.endswith(".obj")]
        )
        merged_mesh = simplify_trimesh(merged_mesh, target_faces=10000)
    except Exception as e:
        print(f"❌ Failed processing mesh for {shape_id}: {e}")
        return 0

    # Random space samples
    pc_random = np.random.uniform(low=-1, high=1, size=(num_random, 3))
    part_labels_random = np.full((num_random,), -1)
    occ_random = np.full((num_random), 0)

    # Combine and compute SDF
    
    points = np.vstack([pc, pc_random])
    
    part_labels = np.concatenate([part_labels_surface, part_labels_random])

    occupancies = np.concatenate([occupancies, occ_random])
    ''''
    print(f"Computing SDF for {shape_id}")
    sdf = compute_sdf(points, merged_mesh)
    print(f"Computed SDF for {shape_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(output_path, np.hstack([points, sdf.reshape(-1, 1), part_labels.reshape(-1, 1)]))
    '''
    output_path = os.path.join(output_dir, f"{shape_id}.npy")
    print(f"here is the shape of occupancies {occupancies.shape}, /nlabel {part_labels.shape}")
    np.save(output_path, np.hstack([points, occupancies.reshape(-1, 1), part_labels.reshape(-1, 1)]))
    print(f"✅ Saved: {output_path}")
    return 1


def load_train_split(split_path):
    with open(split_path, "r") as f:
        split = json.load(f)
    return [entry["anno_id"] for entry in split]

if __name__ == "__main__":
    split_path = "./data/knife/Knife.train.json"
    partnet_dir = "./data/knife"
    output_dir = "./data/knife_preprocessed_split"

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
