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

@hydra.main(
    version_base=None,
    config_path="../configs/overfitting_configs",
    config_name="overfit_plane",
)
def main(cfg: DictConfig):
    #obj = trimesh.load("./meshes/first_mesh.ply")
    dataio.PointCloud(path="./1193/objs/original-5.obj",
                      on_surface_points=10000,
                      cfg=cfg)

if __name__ == "__main__":
    main()