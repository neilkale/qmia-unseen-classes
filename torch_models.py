import os

os.environ["HF_DATASETS_CACHE"] = "./data/huggingface/datasets"

import torch
import torch.nn as nn
import torchvision.models as tvm
import transformers
from peft import LoraConfig, get_peft_model
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
)

# from transformers import AutoModelForImageClassification, AutoFeatureExtractor, ResNetForImageClassification, \
#     ResNetConfig, ViTConfig, ViTForImageClassification
from transformers import (
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoConfig,
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
        elif model == "cifar-wideresnet":
            return WideResNet(num_classes=num_classes)
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

MLP_CONFIGS = {
    "purchase": {"input_dim": 600},
    "texas": {"input_dim": 6169},
}

# Shared architectures that can be applied to any dataset
MLP_ARCHITECTURES = {
    # A linear classifier (no hidden layers)
    "linear": [],
    # A single-layer network for matching the RMIA mlp setting.
    "tiny": [128],
    # A compact, two-layer network for quick baselines
    "small": [256, 128],
    # A balanced, medium-sized default that works well for both datasets
    "medium": [1024, 512, 256],
    # A wider alternative to the default to retain more features at a high dimension
    "wide": [1024, 1024],
    # A deep and high-capacity network for more complex tasks
    "large": [2048, 1024, 512, 256],
}

def get_fresh_tabular_mlp_model(
    architecture, num_classes=10, hidden_dims=[], extra_inputs=None
):
    """
    Creates a tabular MLP model for structured data.
    
    Supports both predefined architectures and custom layer specifications:
    - mlp-purchase-medium: [512, 256, 128, 64] (600 input features)
    - mlp-texas-large: [2048, 1024, 512, 256, 128] (3084 input features)

    - mlp-purchase-{dim1}_{dim2}_...: Custom layer dimensions (600 input features)
    - mlp-texas-{dim1}_{dim2}_...: Custom layer dimensions (3084 input features)
    Examples:
    - mlp-purchase-256_128: [256, 128] (600 input features)
    """

    try:
        parts = architecture.split("-")
        if parts[0] != "mlp" or len(parts) < 2:
            raise ValueError("Format error.")

        dataset_name = parts[1]
        # Use 'default' if no size/dims are specified, otherwise use the third part
        size_key = parts[2] if len(parts) > 2 else "default"
    except (IndexError, ValueError):
        raise ValueError(
            f"Invalid architecture string: '{architecture}'. "
            "Expected format: 'mlp-{dataset}-{size_or_dims}'."
        )

    dataset_config = MLP_CONFIGS.get(dataset_name)
    if not dataset_config:
        raise NotImplementedError(f"Dataset configuration '{dataset_name}' not found.")
    input_dim = dataset_config["input_dim"]

    # 3. Get architecture to determine hidden layer dimensions
    layer_dims = MLP_ARCHITECTURES.get(size_key)
    if layer_dims is None:
        try:
            layer_dims = [int(dim) for dim in size_key.split("_")]
            if not layer_dims: raise ValueError("Empty custom layers.")
        except ValueError:
            raise NotImplementedError(
                f"Architecture '{size_key}' is not predefined and is not a valid custom format (e.g., '512_256')."
            )

    # 4. Build the model
    layers = []
    prev_dim = input_dim
    for hidden_dim in layer_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU()) if len(layer_dims) > 1 else layers.append(nn.Tanh()) # Use Tanh for single hidden-layer networks
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, num_classes))

    return nn.Sequential(*layers)

class HFSeqClsWrapper(nn.Module):
    """
    Wraps a Hugging Face sequence classification model to accept LongTensor input_ids
    and return logits tensor directly, matching the project's expected interface.
    """
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, input_ids: torch.Tensor):
        outputs = self.hf_model(input_ids=input_ids)
        return outputs.logits

class TextEmbeddingBagMLP(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dims: list, num_outputs: int, pad_token_id: int = 50256):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_token_id)
        self.proj = nn.Linear(embed_dim, embed_dim)
        mlp_layers = []
        prev = embed_dim
        for hd in hidden_dims:
            mlp_layers.append(nn.Linear(prev, hd))
            mlp_layers.append(nn.GELU())
            mlp_layers.append(nn.Dropout(0.1))
            prev = hd
        mlp_layers.append(nn.Linear(prev, num_outputs))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, input_ids: torch.Tensor):
        # input_ids: [batch, seq]
        mask = (input_ids != self.pad_token_id).float()  # [b, s]
        emb = self.embedding(input_ids)  # [b, s, d]
        summed = (emb * mask.unsqueeze(-1)).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        pooled = summed / denom
        pooled = self.proj(pooled)
        return self.mlp(pooled)

def get_text_mlp_model(
    size: str = "small",
    num_outputs: int = 2,
    vocab_size: int = 50257,
    pad_token_id: int = 50256,
):
    presets = {
        "tiny": {"embed": 128, "hidden": [128]},
        "small": {"embed": 256, "hidden": [256, 128]},
        "medium": {"embed": 384, "hidden": [384, 192]},
    }
    cfg = presets.get(size)
    if cfg is None:
        # allow custom like text-mlp-256_128
        try:
            h = [int(x) for x in size.split("_")]
            cfg = {"embed": max(h[0] // 2, 64), "hidden": h}
        except Exception:
            raise NotImplementedError(f"Unknown text-mlp size: {size}")
    return TextEmbeddingBagMLP(vocab_size=vocab_size, embed_dim=cfg["embed"], hidden_dims=cfg["hidden"], num_outputs=num_outputs, pad_token_id=pad_token_id)

class DistilGPT2EncoderHead(nn.Module):
    def __init__(self, base_model_name: str = "distilgpt2", num_outputs: int = 2):
        super().__init__()
        # Load transformer backbone for hidden states
        self.backbone = AutoModel.from_pretrained(base_model_name)
        for p in self.backbone.parameters():
            p.requires_grad = False
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_outputs),
        )
        pad_id = getattr(self.backbone.config, "pad_token_id", None)
        eos_id = getattr(self.backbone.config, "eos_token_id", None)
        self.pad_token_id = pad_id if pad_id is not None else (eos_id if eos_id is not None else 50256)

    def forward(self, input_ids: torch.Tensor):
        mask = (input_ids != self.pad_token_id).long()
        out = self.backbone(input_ids=input_ids, attention_mask=mask)
        hidden = out.last_hidden_state  # [b, s, h]
        denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.head(pooled)

def get_gpt2_seqcls_lora_model(
    base_model_name: str = "gpt2",
    num_classes: int = 2,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
):
    # Configure classification head
    config = AutoConfig.from_pretrained(base_model_name, num_labels=num_classes)
    # Ensure pad token handling aligns with dataset padding by EOS when needed
    if getattr(config, "pad_token_id", None) is None and getattr(config, "eos_token_id", None) is not None:
        config.pad_token_id = config.eos_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        config=config,
    )

    # Target LoRA modules typical for GPT-2 blocks
    target_modules = ["c_attn", "c_proj", "c_fc"]

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    return HFSeqClsWrapper(model)

def get_model(
    architecture,
    n_outputs,
    freeze_embedding=False,
    hidden_dims=[],
    extra_inputs=None,
):
    if architecture == "gpt2-seqcls-lora":
        model = get_gpt2_seqcls_lora_model(num_classes=n_outputs)
    elif architecture.startswith("text-mlp"):
        # formats: text-mlp, text-mlp-small, text-mlp-256_128
        parts = architecture.split("-")
        size = parts[2] if len(parts) > 2 else "small"
        model = get_text_mlp_model(size=size, num_outputs=n_outputs)
    elif architecture == "text-distilgpt2":
        model = DistilGPT2EncoderHead(num_outputs=n_outputs)
    elif architecture.startswith("cifar"):
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
    elif architecture.startswith("mlp"):
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
