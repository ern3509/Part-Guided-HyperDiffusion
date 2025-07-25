import numpy as np
import os
import trimesh
from omegaconf import DictConfig
import sys
import hydra

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from siren import dataio
from siren.experiment_scripts import creating_pointclouds

@hydra.main(
    version_base=None,
    config_path="../configs/overfitting_configs",
    config_name="overfit_plane",
)
def main(cfg: DictConfig):
    #obj = trimesh.load("./meshes/first_mesh.ply")
   # dataio.PointCloud(path="./data/knifeparts_overfitt",
                     # on_surface_points=10000,
                     # cfg=cfg)

    knife_full = creating_pointclouds.merge_parts_of_multiple_objects("./knife_4parts")
    #knife_full.export("knife.obj")

    #part_label = np.load("./data/knife_preprocessed_split/1167.npy")
if __name__ == "__main__":
    main()