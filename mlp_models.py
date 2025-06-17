import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as dist
from torch import nn

from embedder import Embedder


class MLP(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_neurons,
        use_tanh=True,
        over_param=False,
        use_bias=True,
    ):
        super().__init__()
        multires = 1
        self.over_param = over_param
        if not over_param:
            self.embedder = Embedder(
                include_input=True,
                input_dims=2,
                max_freq_log2=multires - 1,
                num_freqs=multires,
                log_sampling=True,
                periodic_fns=[torch.sin, torch.cos],
            )
        self.layers = nn.ModuleList([])

        self.layers.append(nn.Linear(in_size, hidden_neurons[0], bias=use_bias))
        for i, _ in enumerate(hidden_neurons[:-1]):
            self.layers.append(
                nn.Linear(hidden_neurons[i], hidden_neurons[i + 1], bias=use_bias)
            )
        self.layers.append(nn.Linear(hidden_neurons[-1], out_size, bias=use_bias))
        self.use_tanh = use_tanh

    def forward(self, x):
        if not self.over_param:
            x = self.embedder.embed(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        if self.use_tanh:
            x = torch.tanh(x)
        return x, None


class MLP3D(nn.Module):
    def __init__(
        self,
        n_of_parts,
        out_size,
        hidden_neurons,
        use_leaky_relu=False,
        use_bias=True,
        multires=10,
        output_type=None,
        move=False,
        semantic = False,
        **kwargs,
    ):
        super().__init__()
        self.embedder = Embedder(
            include_input=True,
            input_dims=3 if not move else 4,
            max_freq_log2=multires - 1,
            num_freqs=multires,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.semantic = semantic
        self.n_of_parts = n_of_parts
        self.layers = nn.ModuleList([])
        self.output_type = output_type
        self.use_leaky_relu = use_leaky_relu
        in_size = self.embedder.out_dim
        self.layers.append(nn.Linear(in_size, hidden_neurons[0], bias=use_bias))

        
        ''''
        self.classification_heads = nn.ModuleList([
            nn.Linear(sum(hidden_neurons), 1)  # Each outputs a single logit
            for _ in range(n_of_parts)
        ])
        '''
        for i, _ in enumerate(hidden_neurons[:-1]):
            self.layers.append(
                nn.Linear(hidden_neurons[i], hidden_neurons[i + 1], bias=use_bias)
            )
        self.layers.append(nn.Linear(hidden_neurons[-1], out_size, bias=use_bias))

        # Classification heads as additional layers
        for i in range(n_of_parts):
            self.layers.append(nn.Linear(sum(hidden_neurons), 1, bias=use_bias))

        self.main_network_depth = len(hidden_neurons) + 1  # +1 for input layer
        self.classification_head_indices = list(
            range(self.main_network_depth, self.main_network_depth + n_of_parts))

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        x = coords_org
        x = self.embedder.embed(x)
        activations = []

        for i, layer in enumerate(self.layers[:self.main_network_depth]):
            x = layer(x)
            if i < self.main_network_depth - 1:
                x = F.leaky_relu(x) if self.use_leaky_relu else F.relu(x)
                if self.semantic:
                    #for class_id in self.classification_head_indices:
                    #    mask = class_mask.get_mask(class_id - self.classification_head_indices[0])   #the mask indices start at 0
                    activations.append(x)
        
        total_neurons = torch.cat(activations, dim=2) if activations[0].dim() == 3 else torch.cat(activations, dim=1) #at inference there is no batching
        if self.output_type == "occ":
            # x = torch.sigmoid(x)
            pass
        elif self.output_type == "sdf":
            x = torch.tanh(x)
        elif self.output_type == "logits":
            x = x
        else:
            raise f"This self.output_type ({self.output_type}) not implemented"
        #x = dist.Bernoulli(logits=x).logits

        if self.semantic:
            # Compute auxiliary output directly from selected activations
            class_mask = ClassMaskCreator(self, self.n_of_parts)
            part_outputs = []
            for class_id in self.classification_head_indices:
                mask = class_mask.get_mask(class_id - self.classification_head_indices[0])   #the mask indices start at 0
                
                masked_activation = mask*total_neurons
                class_output = self.layers[class_id](masked_activation)
                part_outputs.append(class_output.squeeze(-1))
            
            part_outputs = torch.cat(part_outputs, dim=0).T

            return {"model_in": coords_org, "model_out": x, "part_classification":part_outputs}
                

        return {"model_in": coords_org, "model_out": x}

class ClassMaskCreator:
    def __init__(self, model: nn.Module, n_classes: int):
        """
        Initialize the class-wise neuron masks for the MLP.

        Args:
            model (nn.Module): The MLP model
            n_classes (int): Number of semantic part classes
        """
        self.n_classes = n_classes
        self.class_masks = {}

        # Compute total number of neurons across a linear layer
        #Caveat: we assume that as in hyperdiffusion paper the layer have the same number of neurons
        total_neurons = model.layers[0].out_features #should be 128
        ''''
        sum(
            layer.out_features for i, layer in enumerate(model.layers) if isinstance(layer, nn.Linear) and i < 3
        )
        '''
        neurons_per_class = total_neurons // n_classes
        start_idx = 0

        # Create masks for each class
        for class_id in range(n_classes):
            mask = torch.zeros(total_neurons)

            end_idx = start_idx + neurons_per_class
            # Ensure last class gets any remainder
            if class_id == n_classes - 1:
                end_idx = total_neurons

            mask[start_idx:end_idx] = 1.0

            self.class_masks[class_id] = mask
            start_idx = end_idx

    def get_mask(self, class_id: int):
        """
        Retrieve the binary mask for a given class.

        Args:
            class_id (int): The semantic part class ID

        Returns:
            torch.Tensor: Binary mask of shape [total_neurons]
        """
        return self.class_masks[class_id]


class class_to_param_mask_map():
    ''''
    The goal of this function is to assign an additional classification task to blocks of MLP 
    '''
    def __init__(self, model, 
                 n=1 ):#number of part
        self.class_mask = {class_id:{} for class_id in range(n)}
       
        param_dict = dict(model.named_parameters())
        
        for name, param in param_dict:
            shape = param.shape

            for class_id in range(n):
                self.class_masks[class_id][name] = torch.zeros_like(param.data)

            if len(shape) == 2:
                out_features = shape[1] #get the output shape of the layer
                neurons_per_class = out_features // n #choice of  a magic number is justified due to the number of neuron per hidden layer (128)

                for class_id in range(n):
                    start = class_id * neurons_per_class
                    # Ensure last class gets the remainder if not divisible
                    end = (class_id + 1) * neurons_per_class if class_id != n - 1 else out_features
                    # Set those rows to 1 in the class mask
                    self.class_masks[class_id][name][start:end, :] = 1

            elif len(shape) == 1: #bias
                # For 1D tensors like bias (e.g., [out_features])
                length = shape[0]
                elems_per_class = length // n  # split into class chunks

                for class_id in range(n):
                    start = class_id * elems_per_class
                    end = (class_id + 1) * elems_per_class if class_id != n - 1 else length
                    self.class_masks[class_id][name][start:end] = 1  # assign to class

    pass