import diff_operators
import torch
import torch.nn.functional as F
import numpy as np

def image_mse(mask, model_output, gt):
    if mask is None:
        return {"img_loss": ((model_output["model_out"] - gt["img"]) ** 2).mean()}
    else:
        return {
            "img_loss": (mask * (model_output["model_out"] - gt["img"]) ** 2).mean()
        }


def image_l1(mask, model_output, gt):
    if mask is None:
        return {"img_loss": torch.abs(model_output["model_out"] - gt["img"]).mean()}
    else:
        return {
            "img_loss": (mask * torch.abs(model_output["model_out"] - gt["img"])).mean()
        }


def image_mse_TV_prior(mask, k1, model, model_output, gt):
    coords_rand = 2 * (
        torch.rand(
            (
                model_output["model_in"].shape[0],
                model_output["model_in"].shape[1] // 2,
                model_output["model_in"].shape[2],
            )
        ).cuda()
        - 0.5
    )
    rand_input = {"coords": coords_rand}
    rand_output = model(rand_input)

    if mask is None:
        return {
            "img_loss": ((model_output["model_out"] - gt["img"]) ** 2).mean(),
            "prior_loss": k1
            * (
                torch.abs(
                    diff_operators.gradient(
                        rand_output["model_out"], rand_output["model_in"]
                    )
                )
            ).mean(),
        }
    else:
        return {
            "img_loss": (mask * (model_output["model_out"] - gt["img"]) ** 2).mean(),
            "prior_loss": k1
            * (
                torch.abs(
                    diff_operators.gradient(
                        rand_output["model_out"], rand_output["model_in"]
                    )
                )
            ).mean(),
        }


def image_mse_FH_prior(mask, k1, model, model_output, gt):
    coords_rand = 2 * (
        torch.rand(
            (
                model_output["model_in"].shape[0],
                model_output["model_in"].shape[1] // 2,
                model_output["model_in"].shape[2],
            )
        ).cuda()
        - 0.5
    )
    rand_input = {"coords": coords_rand}
    rand_output = model(rand_input)

    img_hessian, status = diff_operators.hessian(
        rand_output["model_out"], rand_output["model_in"]
    )
    img_hessian = img_hessian.view(*img_hessian.shape[0:2], -1)
    hessian_norm = img_hessian.norm(dim=-1, keepdim=True)

    if mask is None:
        return {
            "img_loss": ((model_output["model_out"] - gt["img"]) ** 2).mean(),
            "prior_loss": k1 * (torch.abs(hessian_norm)).mean(),
        }
    else:
        return {
            "img_loss": (mask * (model_output["model_out"] - gt["img"]) ** 2).mean(),
            "prior_loss": k1 * (torch.abs(hessian_norm)).mean(),
        }


def latent_loss(model_output):
    return torch.mean(model_output["latent_vec"] ** 2)


def hypo_weight_loss(model_output):
    weight_sum = 0
    total_weights = 0

    for weight in model_output["hypo_params"].values():
        weight_sum += torch.sum(weight**2)
        total_weights += weight.numel()

    return weight_sum * (1 / total_weights)


def image_hypernetwork_loss(mask, kl, fw, model_output, gt):
    return {
        "img_loss": image_mse(mask, model_output, gt)["img_loss"],
        "latent_loss": kl * latent_loss(model_output),
        "hypo_weight_loss": fw * hypo_weight_loss(model_output),
    }


def function_mse(model_output, gt):
    return {"func_loss": ((model_output["model_out"] - gt["func"]) ** 2).mean()}


def gradients_mse(model_output, gt):
    # compute gradients on the model
    gradients = diff_operators.gradient(
        model_output["model_out"], model_output["model_in"]
    )
    # compare them with the ground-truth
    gradients_loss = torch.mean((gradients - gt["gradients"]).pow(2).sum(-1))
    return {"gradients_loss": gradients_loss}


def gradients_color_mse(model_output, gt):
    # compute gradients on the model
    gradients_r = diff_operators.gradient(
        model_output["model_out"][..., 0], model_output["model_in"]
    )
    gradients_g = diff_operators.gradient(
        model_output["model_out"][..., 1], model_output["model_in"]
    )
    gradients_b = diff_operators.gradient(
        model_output["model_out"][..., 2], model_output["model_in"]
    )
    gradients = torch.cat((gradients_r, gradients_g, gradients_b), dim=-1)
    # compare them with the ground-truth
    weights = torch.tensor([1e1, 1e1, 1.0, 1.0, 1e1, 1e1]).cuda()
    gradients_loss = torch.mean(
        (weights * (gradients[0:2] - gt["gradients"]).pow(2)).sum(-1)
    )
    return {"gradients_loss": gradients_loss}


def laplace_mse(model_output, gt):
    # compute laplacian on the model
    laplace = diff_operators.laplace(
        model_output["model_out"], model_output["model_in"]
    )
    # compare them with the ground truth
    laplace_loss = torch.mean((laplace - gt["laplace"]) ** 2)
    return {"laplace_loss": laplace_loss}


def wave_pml(model_output, gt):
    source_boundary_values = gt["source_boundary_values"]
    x = model_output["model_in"]  # (meta_batch_size, num_points, 3)
    y = model_output["model_out"]  # (meta_batch_size, num_points, 1)
    squared_slowness = gt["squared_slowness"]
    dirichlet_mask = gt["dirichlet_mask"]
    batch_size = x.shape[1]

    du, status = diff_operators.jacobian(y, x)
    dudt = du[..., 0]

    if torch.all(dirichlet_mask):
        diff_constraint_hom = torch.Tensor([0])
    else:
        hess, status = diff_operators.jacobian(du[..., 0, :], x)
        lap = hess[..., 1, 1, None] + hess[..., 2, 2, None]
        dudt2 = hess[..., 0, 0, None]
        diff_constraint_hom = dudt2 - 1 / squared_slowness * lap

    dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]
    neumann = dudt[dirichlet_mask]

    return {
        "dirichlet": torch.abs(dirichlet).sum() * batch_size / 1e1,
        "neumann": torch.abs(neumann).sum() * batch_size / 1e2,
        "diff_constraint_hom": torch.abs(diff_constraint_hom).sum(),
    }


def helmholtz_pml(model_output, gt):
    source_boundary_values = gt["source_boundary_values"]

    if "rec_boundary_values" in gt:
        rec_boundary_values = gt["rec_boundary_values"]

    wavenumber = gt["wavenumber"].float()
    x = model_output["model_in"]  # (meta_batch_size, num_points, 2)
    y = model_output["model_out"]  # (meta_batch_size, num_points, 2)
    squared_slowness = gt["squared_slowness"].repeat(1, 1, y.shape[-1] // 2)
    batch_size = x.shape[1]

    full_waveform_inversion = False
    if "pretrain" in gt:
        pred_squared_slowness = y[:, :, -1] + 1.0
        if torch.all(gt["pretrain"] == -1):
            full_waveform_inversion = True
            pred_squared_slowness = torch.clamp(y[:, :, -1], min=-0.999) + 1.0
            squared_slowness_init = torch.stack(
                (
                    torch.ones_like(pred_squared_slowness),
                    torch.zeros_like(pred_squared_slowness),
                ),
                dim=-1,
            )
            squared_slowness = torch.stack(
                (pred_squared_slowness, torch.zeros_like(pred_squared_slowness)), dim=-1
            )
            squared_slowness = torch.where(
                (torch.abs(x[..., 0, None]) > 0.75)
                | (torch.abs(x[..., 1, None]) > 0.75),
                squared_slowness_init,
                squared_slowness,
            )
        y = y[:, :, :-1]

    du, status = diff_operators.jacobian(y, x)
    dudx1 = du[..., 0]
    dudx2 = du[..., 1]

    a0 = 5.0

    # let pml extend from -1. to -1 + Lpml and 1 - Lpml to 1.0
    Lpml = 0.5
    dist_west = -torch.clamp(x[..., 0] + (1.0 - Lpml), max=0)
    dist_east = torch.clamp(x[..., 0] - (1.0 - Lpml), min=0)
    dist_south = -torch.clamp(x[..., 1] + (1.0 - Lpml), max=0)
    dist_north = torch.clamp(x[..., 1] - (1.0 - Lpml), min=0)

    sx = (
        wavenumber * a0 * ((dist_west / Lpml) ** 2 + (dist_east / Lpml) ** 2)[..., None]
    )
    sy = (
        wavenumber
        * a0
        * ((dist_north / Lpml) ** 2 + (dist_south / Lpml) ** 2)[..., None]
    )

    ex = torch.cat((torch.ones_like(sx), -sx / wavenumber), dim=-1)
    ey = torch.cat((torch.ones_like(sy), -sy / wavenumber), dim=-1)

    A = modules.compl_div(ey, ex).repeat(1, 1, dudx1.shape[-1] // 2)
    B = modules.compl_div(ex, ey).repeat(1, 1, dudx1.shape[-1] // 2)
    C = modules.compl_mul(ex, ey).repeat(1, 1, dudx1.shape[-1] // 2)

    a, _ = diff_operators.jacobian(modules.compl_mul(A, dudx1), x)
    b, _ = diff_operators.jacobian(modules.compl_mul(B, dudx2), x)

    a = a[..., 0]
    b = b[..., 1]
    c = modules.compl_mul(modules.compl_mul(C, squared_slowness), wavenumber**2 * y)

    diff_constraint_hom = a + b + c
    diff_constraint_on = torch.where(
        source_boundary_values != 0.0,
        diff_constraint_hom - source_boundary_values,
        torch.zeros_like(diff_constraint_hom),
    )
    diff_constraint_off = torch.where(
        source_boundary_values == 0.0,
        diff_constraint_hom,
        torch.zeros_like(diff_constraint_hom),
    )
    if full_waveform_inversion:
        data_term = torch.where(
            rec_boundary_values != 0,
            y - rec_boundary_values,
            torch.Tensor([0.0]).cuda(),
        )
    else:
        data_term = torch.Tensor([0.0])

        if "pretrain" in gt:  # we are not trying to solve for velocity
            data_term = pred_squared_slowness - squared_slowness[..., 0]

    return {
        "diff_constraint_on": torch.abs(diff_constraint_on).sum() * batch_size / 1e3,
        "diff_constraint_off": torch.abs(diff_constraint_off).sum(),
        "data_term": torch.abs(data_term).sum() * batch_size / 1,
    }


def occ_sigmoid1(model_output, gt, model, cfg=None, first_state_dict=None):
    gt_sdf = gt["sdf"]
    pred_sdf = model_output["model_out"]
    if cfg.kl_weight > 0 and first_state_dict is not None:
        param_arr = []
        for param in model.parameters():
            param_arr.append(param.flatten())
        param_vec = torch.hstack(param_arr)
        curr_dist = torch.distributions.Normal(
            torch.mean(param_vec), torch.var(param_vec)
        )
        param_arr = []
        for l in first_state_dict:
            param_arr.append(first_state_dict[l].flatten())
        param_vec = torch.hstack(param_arr)
        target_dist = torch.distributions.Normal(
            torch.mean(param_vec), torch.var(param_vec)
        )
        kl_loss = torch.distributions.kl_divergence(curr_dist, target_dist)
        return {
            "occupancy": F.binary_cross_entropy(pred_sdf, gt_sdf),
            "kl_weight": cfg.kl_weight * kl_loss,
        }
    else:
        #Documentation: Weighting of the loss due to an unbalanced distribution of the point cloud
        pos_weight = torch.tensor(4.0).to(pred_sdf.device)
        loss = F.binary_cross_entropy_with_logits(
            pred_sdf.squeeze(-1), gt_sdf.squeeze(-1), reduction="none", pos_weight=pos_weight
        )

    
        # loss = F.binary_cross_entropy_with_logits(pred_sdf.squeeze(-1), gt_sdf.squeeze(-1))
        # return {'occupancy': loss}
        return {"occupancy": loss.sum(-1).mean()}

def occ_sigmoid(model_output, gt, model, cfg=None, first_state_dict=None):
    gt_sdf = gt["sdf"]
    pred_sdf = model_output["model_out"]
    
    #pos_weight = torch.tensor(10.0).to(pred_sdf.device)
    loss = F.binary_cross_entropy_with_logits(
        pred_sdf.squeeze(-1), gt_sdf.squeeze(-1), reduction="none"
    )   
    
    if cfg.kl_weight > 0 and first_state_dict is not None:
        param_arr = [p.flatten() for p in model.parameters()]
        curr_vec = torch.hstack(param_arr)
        curr_dist = torch.distributions.Normal(
            torch.mean(curr_vec), torch.var(curr_vec)
        )

        param_arr = [first_state_dict[k].flatten() for k in first_state_dict]
        ref_vec = torch.hstack(param_arr)
        ref_dist = torch.distributions.Normal(
            torch.mean(ref_vec), torch.var(ref_vec)
        )

        kl = torch.distributions.kl_divergence(curr_dist, ref_dist)
        return {
            "occupancy": loss,
            "kl_weight": cfg.kl_weight * kl
        }
    
    return {"occupancy": loss}

def occ_sigmoid_semantic(model_output, gt, model, cfg=None, first_state_dict=None):
    gt_sdf = gt["sdf"]
    pred_sdf = model_output["model_out"]
    pred_label = model_output["part_classification"]
    gt_label = gt["labels"].squeeze(0)

    semantic_effort = 0.3 # 0.5
    #pos_weight = torch.tensor(10.0).to(pred_sdf.device)
    occ_loss = F.binary_cross_entropy_with_logits(
        pred_sdf.squeeze(-1), gt_sdf.squeeze(-1), reduction="none"
    )   
    losses = {}
    n_parts = cfg.multi_process.n_of_parts
    
    # Occupancy loss (shared)
    occ_loss = F.binary_cross_entropy_with_logits(pred_sdf, gt_sdf)
    losses['occupancy'] = occ_loss

    # Create one-hot target for part labels: [B, n_parts]
    
    gt_one_hot = to_one_hot(gt_label, num_classes=n_parts)
    # Classification loss: each block learns to predict 1 if point belongs to it, 0 otherwise
    cls_loss_all = F.binary_cross_entropy_with_logits(pred_label, gt_one_hot, reduction='none')  # [B, n_parts]
    #print(cls_loss_all.shape)
    mean_cls_loss = cls_loss_all.mean()
    losses['semantic_mean'] = mean_cls_loss
    for part_id in range(n_parts):
        cls_loss_part = cls_loss_all[:, part_id].mean() 
        losses[f'block_{part_id}'] = semantic_effort * cls_loss_part + occ_loss  # you can scale this if needed

    losses['total'] = semantic_effort * mean_cls_loss + occ_loss
    #print(losses.items())
    return losses

def to_one_hot(indices, num_classes):
    """
    Converts a 1D array of class indices to one-hot encoded 2D array.

    Args:
        indices (np.ndarray): Array of shape (N,) with integer class indices.
        num_classes (int): Total number of classes.

    Returns:
        np.ndarray: One-hot encoded array of shape (N, num_classes)
    """

    #squeeze the input
    indices = indices.squeeze(axis=-1)
    indices = np.asarray(indices).astype(int)
    one_hot = np.zeros((indices.size, num_classes), dtype=np.float32)
    
   # print("Max index:", indices.max(), "Num classes:", num_classes)

    # Mask for valid indices (not -1)
    valid_mask = indices != -1
    valid_indices = indices[valid_mask]

    # Apply one-hot only to valid positions
    one_hot[valid_mask, valid_indices] = 1.0
    return torch.from_numpy(one_hot.astype(float)) 

def occ_tanh(model_output, gt, model):
    gt_sdf = gt["sdf"]
    pred_sdf = model_output["model_out"]
    # kl_loss_fn = torchbnn.BKLLoss(reduction='mean', last_layer_only=False)
    # kl_loss = kl_loss_fn(model)
    return {"occupancy": F.mse_loss(pred_sdf, gt_sdf)}  # , 'kl_weights': 0.1 * kl_loss}


def sdf(model_output, gt, model, cfg=None):
    """
    x: batch of input coordinates
    y: usually the output of the trial_soln function
    """
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]
    # kl_loss_fn = torchbnn.BKLLoss(reduction='mean', last_layer_only=False)
    coords = model_output["model_in"]
    pred_sdf = model_output["model_out"]

    gradient = diff_operators.gradient(pred_sdf, coords)
    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(
        gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf))
    )
    normal_constraint = torch.where(
        gt_sdf != -1,
        1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
        torch.zeros_like(gradient[..., :1]),
    )
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
    # kl_loss = kl_loss_fn(model)
    # Exp      # Lapl
    # -----------------
    return {
        "sdf": torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
        "inter": inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
        "normal_constraint": normal_constraint.mean() * 1e2,  # 1e2
        "grad_constraint": grad_constraint.mean() * 5e1,
    }  # 1e1      # 5e1


# inter = 3e3 for ReLU-PE

def sdf_and_part_classification(model_output, gt, model , cfg= None):
    pass