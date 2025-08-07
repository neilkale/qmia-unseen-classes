import os

os.environ["HF_DATASETS_CACHE"] = "./data/huggingface/datasets"

import torch
import torchvision.models as tvm
import transformers
from cifar_architectures import (
    ResNet10,
    ResNet10ExtraInputs,
    ResNet18,
    ResNet18ExtraInputs,
    ResNet34,
    ResNet34ExtraInputs,
    ResNet50,
    ResNet50ExtraInputs,
    WideResNet,
    ResNet10DP,
    ResNet18DP,
    ResNet34DP,
    ResNet50DP,
    ResNet101DP,
    ResNet152DP,
    ResNet50NoNorm,
)

# from transformers import AutoModelForImageClassification, AutoFeatureExtractor, ResNetForImageClassification, \
#     ResNetConfig, ViTConfig, ViTForImageClassification
from transformers import (
    AutoModelForImageClassification,
    ResNetConfig,
    ViTConfig,
    ViTForImageClassification,
)

RESNET18CONFIG = ResNetConfig(
    depths=[2, 2, 2, 2],
    downsample_in_first_stage=False,
    embedding_size=64,
    hidden_act="relu",
    hidden_sizes=[16, 32, 64],
    layer_type="basic",
    num_channels=3,
)


RESNET50CONFIG = ResNetConfig(
    depths=[3, 4, 6, 3],
    # downsample_in_first_stage=False,
    downsample_in_first_stage=True,
    embedding_size=64,
    hidden_act="relu",
    hidden_sizes=[256, 512, 1024, 2048],
    layer_type="bottleneck",
    num_channels=3,
)

transformers.logging.set_verbosity_error()


class HugginFaceTupleWrapper(torch.nn.Module):
    def __init__(self, model_base, hidden_dims=[], extra_inputs=None):
        super().__init__()
        self.model_base = model_base

        # Replaces the linear layer of the default classifier with an MLP
        if isinstance(self.model_base.classifier, torch.nn.Sequential):
            self.classifier = self.model_base.classifier
            self.model_base.classifier = torch.nn.Identity()
        else:
            prev_size = self.model_base.classifier.in_features
            if extra_inputs is not None:
                prev_size += extra_inputs
            num_classes = self.model_base.classifier.out_features
            mlp_list = []
            for hd in hidden_dims:
                mlp_list.append(torch.nn.Linear(prev_size, hd))
                mlp_list.append(torch.nn.LeakyReLU())  # TODO!
                prev_size = hd
            mlp_list.append(torch.nn.Linear(prev_size, num_classes))
            self.classifier = torch.nn.Sequential(*mlp_list)
            self.model_base.classifier = torch.nn.Identity()

        # self.linear =torch.nn.Linear(embedding_size, num_classes)
        super(HugginFaceTupleWrapper, self).add_module("model_base", self.model_base)
        super(HugginFaceTupleWrapper, self).add_module("classifier", self.classifier)

    def forward(self, input, extra_inputs=None):
        embedding = self.model_base(input).logits
        if extra_inputs is not None:
            assert (
                extra_inputs.shape[0] == embedding.shape[0]
                and extra_inputs.ndim == embedding.ndim
            ), "extra inputs and embedding need to have the same batch dimension"
            embedding = torch.concatenate([embedding, extra_inputs], dim=1)
            # print(embedding.shape)
        logits = self.classifier(embedding)
        return logits

    def freeze_base_model(self):
        for p in self.model_base.parameters():
            p.requires_grad = False
        for p in self.model_base.classifier.parameters():
            p.requires_grad = True

    def unfreeze_base_model(self):
        for p in self.model_base.parameters():
            p.requires_grad = True


def get_huggingface_model(
    model_checkpoint, num_classes=10, hidden_dims=[], extra_inputs=None
):
    if model_checkpoint.startswith("base"):
        configuration = ViTConfig(num_labels=num_classes)
        model_base = ViTForImageClassification(configuration)
        # feature_extractor = None
    else:
        if model_checkpoint.startswith("/"):
            model_checkpoint = model_checkpoint[1:]
        model_base = AutoModelForImageClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    model = HugginFaceTupleWrapper(
        model_base, hidden_dims=hidden_dims, extra_inputs=extra_inputs
    )

    return model


def get_torchvision_model(
    model_name="convnext-tiny", num_classes=10, sample_input=None, hidden_dims=[]
):
    model = None
    if model_name == "convnext-tiny":
        model_fn = tvm.convnext_tiny
        model_weights = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = model_fn(weights=model_weights)
        
        # Get the actual feature dimension instead of hardcoding
        feature_dim = model.classifier[-1].in_features  # This will be 768 or 1536
        print
        if len(hidden_dims):
            prev_size = feature_dim
            mlp_list = []
            for hd in hidden_dims:
                mlp_list.append(torch.nn.Linear(prev_size, hd))
                mlp_list.append(torch.nn.ReLU())
                prev_size = hd
            mlp_list.append(torch.nn.Linear(prev_size, num_classes))
            model.classifier = torch.nn.Sequential(*mlp_list)
        else:
            model.classifier[-1] = torch.nn.Linear(
                in_features=feature_dim, out_features=num_classes, bias=True
            )
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    return model


def get_fresh_resnet_model(
    model="resnet-18", num_classes=10, hidden_dims=[], extra_inputs=None
):
    if model == "resnet-18":
        model_base_config = RESNET18CONFIG

    elif model == "resnet-50":
        model_base_config = RESNET50CONFIG

    else:
        raise NotImplementedError
    model_base_config.num_labels = num_classes
    model_base = AutoModelForImageClassification.from_config(model_base_config)
    model = HugginFaceTupleWrapper(
        model_base, hidden_dims=hidden_dims, extra_inputs=extra_inputs
    )

    # feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
    # return model, feature_extractor
    return model


def get_cifar_resnet_model(
    model="cifar-resnet-18", num_classes=10, hidden_dims=[], extra_inputs=None
):
    if extra_inputs is None:
        if model == "cifar-resnet-18":
            return ResNet18(num_classes=num_classes)
        elif model == "cifar-resnet-10":
            return ResNet10(num_classes=num_classes)
        elif model == "cifar-resnet-34":
            return ResNet34(num_classes=num_classes)
        elif model == "cifar-resnet-50":
            return ResNet50(num_classes=num_classes)
        elif model == "cifar-resnet-10-dp":
            return ResNet10DP(num_classes=num_classes)
        elif model == "cifar-resnet-18-dp":
            return ResNet18DP(num_classes=num_classes)
        elif model == "cifar-resnet-34-dp":
            return ResNet34DP(num_classes=num_classes)
        elif model == "cifar-resnet-50-dp":
            return ResNet50DP(num_classes=num_classes)
        elif model == "cifar-resnet-50-nonorm":
            return ResNet50NoNorm(num_classes=num_classes)
        elif model == "cifar-wideresnet":
            return WideResNet(num_classes=num_classes)
        elif model == "cifar-wideresnet-dp":
            # DP-compatible WideResNet using GroupNorm instead of BatchNorm
            import functools
            # Use GroupNorm with 8 groups as a BatchNorm replacement for DP
            group_norm_fn = functools.partial(torch.nn.GroupNorm, 8)
            return WideResNet(num_classes=num_classes, bn=group_norm_fn)
        elif model == "cifar-vit":
            # Configure a small ViT model for 32x32 images
            vit_config = ViTConfig(
                image_size=32,
                patch_size=4,
                num_channels=3,
                hidden_size=192,
                num_hidden_layers=6,
                num_attention_heads=3,
                intermediate_size=768,
                num_labels=num_classes,
            )
            vit_model = ViTForImageClassification(config=vit_config)
            return ViTLogitsWrapper(vit_model)  # Wrap to extract logits
    else:
        if model == "cifar-resnet-18":
            return ResNet18ExtraInputs(
                num_classes=num_classes,
                hidden_dims=hidden_dims,
                extra_inputs=extra_inputs,
            )
        elif model == "cifar-resnet-10":
            return ResNet10ExtraInputs(
                num_classes=num_classes,
                hidden_dims=hidden_dims,
                extra_inputs=extra_inputs,
            )
        elif model == "cifar-resnet-34":
            return ResNet34ExtraInputs(
                num_classes=num_classes,
                hidden_dims=hidden_dims,
                extra_inputs=extra_inputs,
            )
        elif model == "cifar-resnet-50":
            return ResNet50ExtraInputs(
                num_classes=num_classes,
                hidden_dims=hidden_dims,
                extra_inputs=extra_inputs,
            )

    raise NotImplementedError

def get_fresh_vit_model(model="vit", num_classes=10):
    """
    Creates a standard ViT model from scratch using a config.
    """
    if model == "vit":
        # Standard ViT-Base configuration
        config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            num_labels=num_classes
        )
    else:
        raise NotImplementedError(f"Fresh ViT model '{model}' not implemented")

    # Create a new ViT model from scratch using the configuration
    model = ViTForImageClassification(config=config)
    return model

class ViTLogitsWrapper(torch.nn.Module):
    """Wrapper for ViT models to extract logits from ImageClassifierOutput"""
    def __init__(self, vit_model):
        super().__init__()
        self.vit_model = vit_model
    
    def forward(self, x):
        output = self.vit_model(x)
        return output.logits

def get_fresh_tabular_mlp_model(
    architecture, num_classes=10, hidden_dims=[], extra_inputs=None
):
    """
    Creates a tabular MLP model for structured data.
    
    Supports both predefined architectures and custom layer specifications:
    - tabular-purchase-mlp: [512, 256, 128, 64]
    - tabular-purchase-mlp-small: [64, 64] 
    - tabular-purchase-mlp-large: [2048, 1024, 512, 256, 128]
    - tabular-purchase-mlp-{dim1}_{dim2}_...: Custom layer dimensions
    
    Examples:
    - tabular-purchase-mlp-256_128: [256, 128]
    - tabular-purchase-mlp-512_512_256: [512, 512, 256]
    """
    
    if architecture == "tabular-purchase-mlp":
        input_dim = 600
        layer_dims = [512, 256, 128, 64]
    elif architecture == "tabular-purchase-mlp-wide":
        input_dim = 600
        layer_dims = [512, 512]
    elif architecture == "tabular-purchase-mlp-small":
        input_dim = 600
        layer_dims = [64, 64]
    elif architecture == "tabular-purchase-mlp-large":
        input_dim = 600
        layer_dims = [2048, 1024, 512, 256, 128]
    elif architecture == "tabular-purchase-mlp-largedeep":
        input_dim = 600
        layer_dims = [1024, 768, 768, 512, 512, 384, 384, 256, 128]
    elif architecture == "tabular-purchase-mlp-xlarge":
        input_dim = 600
        layer_dims = [2048, 2048, 1024, 1024, 512, 256]
    elif architecture == "tabular-purchase-mlp-xxlarge":
        input_dim = 600
        layer_dims = [4096, 4096, 2048, 2048, 1024, 512]
    elif architecture.startswith("tabular-purchase-mlp-") and "_" in architecture:
        # Parse custom layer dimensions from architecture name
        # e.g., "tabular-purchase-mlp-128_128" -> [128, 128]
        input_dim = 600
        try:
            dims_str = architecture.replace("tabular-purchase-mlp-", "")
            layer_dims = [int(dim) for dim in dims_str.split("_")]
            if not layer_dims:
                raise ValueError("No layer dimensions specified")
        except ValueError as e:
            raise ValueError(f"Invalid layer dimensions in '{architecture}': {e}")
    else:
        raise NotImplementedError(f"Tabular model '{architecture}' not implemented")
    
    layers = []
    prev_dim = input_dim
    
    # Add hidden layers with ReLU activation
    for dim in layer_dims:
        layers.append(torch.nn.Linear(prev_dim, dim))
        layers.append(torch.nn.ReLU())
        prev_dim = dim
    
    # Add output layer (no activation)
    layers.append(torch.nn.Linear(prev_dim, num_classes))
    
    model = torch.nn.Sequential(*layers)
    return model

def get_model(
    architecture,
    n_outputs,
    freeze_embedding=False,
    hidden_dims=[],
    extra_inputs=None,
):
    if architecture.startswith("cifar"):
        model = get_cifar_resnet_model(
            architecture,
            num_classes=n_outputs,
            hidden_dims=hidden_dims,
            extra_inputs=extra_inputs,
        )
    elif architecture.startswith("resnet"):
        model = get_fresh_resnet_model(
            architecture,
            num_classes=n_outputs,
            hidden_dims=hidden_dims,
            extra_inputs=extra_inputs,
        )
    elif architecture.startswith("vit"):
        model = get_fresh_vit_model(
            model=architecture, num_classes=n_outputs
        )
    elif architecture.startswith("tabular"):
        model = get_fresh_tabular_mlp_model(
            architecture,
            num_classes=n_outputs,
            hidden_dims=hidden_dims,
            extra_inputs=extra_inputs,
        )

    elif "/" in architecture:
        model = get_huggingface_model(
            architecture,
            num_classes=n_outputs,
            hidden_dims=hidden_dims,
            extra_inputs=extra_inputs,
        )
        if freeze_embedding:
            model.freeze_base_model()
    else:
        model = get_torchvision_model(
            model_name=architecture, num_classes=n_outputs, hidden_dims=hidden_dims
        )

    return model
