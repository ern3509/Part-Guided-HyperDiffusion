
# Part-Guided-HyperDiffusion

Our work adresses the lack of semantic understanding of the 3D Object diffusion based generation models, in paarticular our baseline: Hyperdiffusion.
We thus introduce a part-awarness directly in the generation process.


## Method Overview

![Overview image](/static/overview.svg)
![The MLP Overfitting process has been modified in order to achieve instance semantic comprehension of the 3D objects](/static/introimage.pdf)

## Dependencies

* Python 3.7
* PyTorch 1.13.0
* CUDA 11.7
* Weights & Biases (We heavily rely on it for visualization and monitoring)


For full list please see [hyperdiffusion_env.yaml file](/hyperdiffusion_env.yaml)

## Data
We impleted our model on the partNet Dataset. Please request access right to the dataset ad extract the meshes you'll want to overfitt your model with.
If you want to train the diffusion model, you can find some checkpoints here https://github.com/ern3509/Part-Guided-HyperDiffusion/tree/85a5bc1bf3068467f17eb0e5c6e6fbecfd6a9089/logs/insert_name_here.
Our model has been trained on knives but can also be trained for other object of the ShapeNet core.


## Get Started
You can run the google colab cells we created in the notebook Hyperdiffusion.ipynb
This code is simple and is also a goog introduction to the topic. That should be your entry points along our report.

to start evaluating,
```commandline
python main.py --config-name=train_knife mode=test best_model_save_path=<path/to/checkpoint>
```
### Training
To start training,
```commandline
python main.py --config-name=train_knife
```

We are using [hydra](https://hydra.cc/), you can either specify parameters from corresponding yaml file or directly modify
them from terminal. For instance, to change the number of epochs:

```commandline
python main.py --config-name=train_plane epochs=100 
```
### Overfitting
We already provide overfitted shapes but if you want to do it yourself make sure that you put downloaded [PartNet]([https://shapenet.org/](https://huggingface.co/datasets/ShapeNet/PartNet-archive)) shapes (we applied [ManifoldPlus](https://github.com/hjwdzh/ManifoldPlus) pre-processing) into **data** folder.
After that, we first create point clouds and then start overfitting on those point clouds; following lines do exactly that:
```commandline
python siren/experiment_scripts/train_sdf.py --config-name=overfit_knife strategy=save_pc
python siren/experiment_scripts/train_sdf.py --config-name=overfit_knife
```

## Code Map
### Directories
- **configs**: Containing training and overfitting configs.
- **data**: Downloaded point cloud files including train-val-test splits go here (see [Get Started](#get-started)) 
- **diffusion**: Contains all the diffusion logic. Borrowed from [OpenAI](https://github.com/openai/guided-diffusion) .
- **ldm**: Latent diffusion codebase for Voxel baseline. Borrowed from [official LDM repo](https://github.com/CompVis/latent-diffusion).
- **siren**: Modified [SIREN](https://github.com/vsitzmann/siren) codebase. Includes shape overfitting logic.
- **static**: Images for README file.
- **Pointnet_Pointnet2_pytorch**: Includes Pointnet2 definition and weights for 3D FID calculation.
### Generated Directories
- **lightning_checkpoints**: This will be created once you start training for the first time. It will include checkpoints of the diffusion model, the sub-folder names will be the unique name assigned by the Weights & Biases in addition to timestamp.
- **outputs**: Hydra creates this folder to store the configs but we mainly send our outputs to Weights & Biases, so, it's not that special.
- **orig_meshes**: Here we put generated weights as .pth and sometimes generated meshes.
- **wandb**: Weights & Biases will create this folder to store outputs before sending them to server.
### Files
**Utils**
- **augment.py**: Including some augmentation methods, though we don't use them in the main paper.
- **dataset.py**: `WeightDataset` and `VoxelDataset` definitions which are `torch.Dataset` descendants. Former one is related to our HyperDiffusion method, while the latter one is for Voxel baseline.
- **hd_utils.py**: Many utility methods ranging from rendering to flattening MLP weights.

**Evaluation**

- **torchmetrics_fid.py**: Modified torchmetrics fid implementation to calculate 3D-FID.
- **evaluation_metrics_3d.py**: Methods to calculate MMD, COV and 1-NN from [DPC](https://github.com/luost26/diffusion-point-cloud). Both for 3D and 4D.

**Entry Point**
- **hyyperdiffusion.ipynb**: General Entry point. It is a colab notebook in case you don't have a GPU enough big to train the diffusion model
- **hyperdiffusion_env.yaml**: Conda environment file (see [Get Started](#get-started) section).
- **main.py**: Entry point of our codebase.


**Models**
 
- **mlp_models.py**: Definition of ReLU MLPs with positional encoding.
- **transformer.py**: GPT definition from [G.pt paper](https://github.com/wpeebles/G.pt).
- **embedder.py**: Positional encoding definition.
- **hyperdiffusion.py**: Definition of our method, it includes training, testing and validation logics in the form of a Pytorch Lightning module.


## Acknowledgments

We mainly use the Hyperdiffusion code. [Hyperdiffusion](https://github.com/Rgtemze/HyperDiffusion)
The ground codebases are [SIREN](https://github.com/vsitzmann/siren) and [G.pt](https://github.com/wpeebles/G.pt) papers to build our repository. We also referred to [DPC](https://github.com/luost26/diffusion-point-cloud) for codes like evaluation metrics. We used [OpenAI Guided Diffusion](https://github.com/openai/guided-diffusion) as our diffusion backbone. [LDM](https://github.com/CompVis/latent-diffusion) codebase was useful for us to implement our voxel baseline.

