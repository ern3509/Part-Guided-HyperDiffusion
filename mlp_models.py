import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as dist
from torch import nn

from embedder import Embedder

#remove the part loss and change part layer to modullist to register it
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


class MLP3D_nope(nn.Module):
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
        self.class_mask = ClassMaskCreator(self, self.n_of_parts)


        
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
        masked_activation = x

        for i, layer in enumerate(self.layers[:self.main_network_depth]):
            x = layer(x)
            if i < self.main_network_depth - 1:
                x = F.leaky_relu(x) if self.use_leaky_relu else F.relu(x)
                if self.semantic:
                    for class_id in self.classification_head_indices:
                        mask = self.class_mask.get_mask(class_id)   #the mask indices start at 0
                        masked_activation = layer(masked_activation)
                    activations.append(x)
        if self.semantic:
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


class MLP3D_final(nn.Module):
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
        self.layers = nn.ModuleList([
            nn.Linear(self.embedder.out_dim, hidden_neurons[0], bias=use_bias),
            *[
                nn.Linear(hidden_neurons[i], hidden_neurons[i + 1], bias=use_bias)
                for i in range(len(hidden_neurons) - 1)
            ],
        ])
        self.output_type = output_type
        self.use_leaky_relu = use_leaky_relu
        in_size = self.embedder.out_dim
        #self.layers.append(nn.Linear(in_size, hidden_neurons[0], bias=use_bias))
        # self.layer_parts = []
        self.part_dim = hidden_neurons[0] // self.n_of_parts  #num of neurons per part
        
        ''''
        self.classification_heads = nn.ModuleList([
            nn.Linear(sum(hidden_neurons), 1)  # Each outputs a single logit
            for _ in range(n_of_parts)
        ])
        '''
        
        # for i, h_dim in enumerate(hidden_neurons[:-1]):
        #     # Split neurons evenly (adjust if uneven division)
        #     layer_part = nn.Linear(in_size, h_dim)
        #     self.layer_parts.append(layer_part)
        #     in_size = h_dim  # Concatenated output becomes next input
        
            # Occupancy head (operates on full concatenated features)
        self.occ_head = nn.Linear(hidden_neurons[-1], out_size)
            
            # Classification heads (one per part)
        self.class_heads = nn.ModuleList([
            nn.Linear(hidden_neurons[-1], 1) for _ in range(n_of_parts)
        ])

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        x = coords_org
        x = self.embedder.embed(x)
        # Store per-part activations for classification
        part_features = []
        masks = ClassMaskCreator(self, n_classes=self.n_of_parts)

        x = self.layers[0](x)
        x_occ = x  # Occupancy output will be based on this first layer
        part_outputs = []
        temporal_part_output = []
        for i, layer_part in enumerate(self.layers[1:]):
            # Process each part's neurons independently
            x_occ = layer_part(x_occ)  # Process through part-specific weights
            x_occ = F.leaky_relu(x_occ) if self.use_leaky_relu else F.relu(x_occ)
            for part_id in range(self.n_of_parts): 
                x_part = temporal_part_output[i * part_id] if i > 0 else x #part_outputs[part_id] if i > 0 else x
                x_part = layer_part(x_part * masks.get_mask(part_id))
                x_part = F.leaky_relu(x_part) if self.use_leaky_relu else F.relu(x_part)
                temporal_part_output.append(x_part)
            part_outputs = temporal_part_output
            # Store last layer's part features for classification

            if i == len(self.layers) - 2:
                part_features = part_outputs  # List of [B, part_dim] tensors
        
        # Occupancy output (full concatenated features)
        occ_output = self.occ_head(x_occ)

        
        # Part classification (only from their dedicated neurons)
        class_outputs = []
        for part_id in range(self.n_of_parts):
            class_out = self.class_heads[part_id](part_features[part_id] * masks.get_mask(part_id))
            class_outputs.append(class_out.squeeze(0))
        
        return {
            "model_in": coords_org, "model_out": occ_output, "part_classification":torch.cat(class_outputs, dim=-1)
        }
    

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
    

class MLP3D_simple_partguided(nn.Module):
    def __init__(
        self,
        n_of_parts=4,  # 4 input layers
        out_size=1,    # 1 output (occupancy)
        hidden_neurons=[64, 64, 64],  # 3 hidden layers
        use_leaky_relu=False,
        use_bias=True,
        multires=10,
        output_type=None,
        move=False,
        semantic=False,
        **kwargs
    ):
        super().__init__()
        
        # Positional embedding
        self.embedder = Embedder(
            include_input=True,
            input_dims=3 if not move else 4,
            max_freq_log2=multires - 1,
            num_freqs=multires,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        
        self.n_of_parts = n_of_parts
        self.output_type = output_type
        self.use_leaky_relu = use_leaky_relu
        self.semantic = semantic
        
        # Input dimensions
        in_size = self.embedder.out_dim
        
        # Create 4 parallel input layers
        self.input_layers = nn.ModuleList()
        self.part_dim = hidden_neurons[0] // n_of_parts
        
        for _ in range(n_of_parts):
            # Each input layer outputs hidden_neurons[0]/4 features
            self.input_layers.append(
                nn.Linear(in_size, self.part_dim, bias=use_bias)
            )
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_neurons)):
            # First hidden layer takes concatenated input from all 4 input layers
            in_features = hidden_neurons[0] if i == 0 else hidden_neurons[i-1]
            self.hidden_layers.append(
                nn.Linear(in_features, hidden_neurons[i], bias=use_bias)
            )
        
        # Output layer (occupancy)
        self.occ_head = nn.Linear(hidden_neurons[-1], out_size)
        
        # Classification heads for each part
        self.class_heads = nn.ModuleList([
            nn.Linear(self.part_dim, 1) for _ in range(n_of_parts)
        ])

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        x = coords_org
        x = self.embedder.embed(x)
        
        # Process through 4 parallel input layers
        part_features = []
        for i, layer in enumerate(self.input_layers):
            part_feat = layer(x)
            part_feat = F.leaky_relu(part_feat) if self.use_leaky_relu else F.relu(part_feat)
            part_features.append(part_feat)
        
        # Get classification outputs (from each part's features)
        class_outputs = []
        for i, (feat, head) in enumerate(zip(part_features, self.class_heads)):
            class_outputs.append(head(feat))
        
        # Concatenate all part features
        x = torch.cat(part_features, dim=-1)
        
        # Process through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.leaky_relu(x) if self.use_leaky_relu else F.relu(x)
        
        # Final occupancy output
        occ_output = self.occ_head(x)
        
        return {
            "model_in": coords_org,
            "model_out": occ_output,
            "part_classification": torch.cat(class_outputs, dim=-1).squeeze()
        }

class MLP3D(nn.Module):
    def __init__(
        self,
        n_of_parts,  # 4 input layers
        out_size,
        hidden_neurons,
        use_leaky_relu=False,
        use_bias=True,
        multires=10,
        output_type=None,
        move=False,
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
        self.layers = nn.ModuleList([])
        self.output_type = output_type
        self.use_leaky_relu = use_leaky_relu
        in_size = self.embedder.out_dim
        self.layers.append(nn.Linear(in_size, hidden_neurons[0], bias=use_bias))
        for i, _ in enumerate(hidden_neurons[:-1]):
            self.layers.append(
                nn.Linear(hidden_neurons[i], hidden_neurons[i + 1], bias=use_bias)
            )
        self.layers.append(nn.Linear(hidden_neurons[-1], out_size, bias=use_bias))

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        x = coords_org
        x = self.embedder.embed(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.leaky_relu(x) if self.use_leaky_relu else F.relu(x)
        x = self.layers[-1](x)

        if self.output_type == "occ":
            # x = torch.sigmoid(x)
            pass
        elif self.output_type == "sdf":
            x = torch.tanh(x)
        elif self.output_type == "logits":
            x = x
        else:
            raise f"This self.output_type ({self.output_type}) not implemented"
        x = dist.Bernoulli(logits=x).logits

        return {"model_in": coords_org, "model_out": x}