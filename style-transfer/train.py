import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from torchvision.utils import save_image

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

PROJECT_DIR = Path(__file__).parent

CONTENT_PATH = PROJECT_DIR / "content.jpg"
STYLE_PATH = PROJECT_DIR / "style.jpg"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok = True)

OUTPUT_IMAGE_PATH = OUTPUT_DIR / "output.png"

#hyperparameters
TV_WEIGHT = 1e-3
IMAGE_SIZE = 512
CONTENT_WEIGHTS = 1.0
STYLE_WEIGHTS = 5_000_000.0
NUM_STEPS = 800

CONTENT_LAYER = "conv4_2"
STYLE_LAYER_WEIGHTS = {
    "conv1_1": 1.0,
    "conv2_1": 0.8,
    "conv3_1": 0.5,
    "conv4_1": 0.2,
    "conv5_1": 0.1,
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

loader = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def load_image(image_path:Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device)

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device= tensor.device).view(1,3,1,1)
    std = torch.tensor(IMAGENET_STD, device= tensor.device).view(1,3,1,1)
    return tensor * std + mean

def save_tensor_image(tensor: torch.Tensor, output_path: Path) -> None:
    image = tensor.detach().cpu().clone()
    image = denormalize(image)
    image = torch.clamp(image, 0.0, 1.0)
    save_image(image.squeeze(0), output_path)

def replace_max_pool_with_avg_pool(model : nn.Sequential) -> nn.Sequential:
    new_layers = []
    for layer in model:
        if isinstance(layer, nn.MaxPool2d):
            new_layers.append(nn.AvgPool2d(kernel_size=layer.kernel_size, stride = layer.stride, padding = layer.padding))
        else:
            new_layers.append(layer)
    return nn.Sequential(*new_layers)

def build_vgg_features() -> nn.Sequential :
    weights = VGG19_Weights.IMAGENET1K_V1
    model = vgg19(weights=weights).features
    model = replace_max_pool_with_avg_pool(model)
    model = model.to(device).eval()

    for param in model.parameters():
        param.requires_grad = False
    
    return model

def get_features(image: torch.Tensor, model: nn.Sequential) -> dict[str, torch.tensor]:
    features = {}
    x = image

    layer_name_map = {
        "0" : "conv1_1",
        "2" : "conv1_2",
        "5" : "conv2_1",
        "7" : "conv2_2",
        "10" : "conv3_1",
        "12" : "conv3_2",
        "14" : "conv3_3",
        "16" : "conv3_4",
        "19" : "conv4_1",
        "21" : "conv4_2",
        "23" : "conv4_3",
        "25" : "conv4_4",
        "28" : "conv5_1",
        "30" : "conv5_2",
        "32" : "conv5_3",
        "34" : "conv5_4",

    }

    for name, layer in model._modules.items():
        x = layer(x)
        if name in layer_name_map:
            features[layer_name_map[name]] = x
    return features

def gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = feature_map.shape
    features = feature_map.view(batch_size,channels,height*width)
    gram = torch.bmm(features, features.transpose(1,2))
    gram = gram/(channels*height*width)
    return gram

def run_style_transfer():
    content_image = load_image(CONTENT_PATH)
    style_image = load_image(STYLE_PATH)

    model = build_vgg_features()

    content_features = get_features(content_image, model)
    style_features = get_features(style_image, model)

    content_target = content_features[CONTENT_LAYER].detach()
    style_targets = {
        layer : gram_matrix(style_features[layer]).detach()
        for layer in STYLE_LAYER_WEIGHTS
    }

    generated = content_image.clone().to(device)
    generated.requires_grad_(True)

    optimizer = optim.LBFGS([generated])

    step = [0]

    while step[0] <=NUM_STEPS:
        def closure():
            optimizer.zero_grad()

            generated_features = get_features(generated, model)

            content_loss = torch.mean(
                (generated_features[CONTENT_LAYER] - content_target) ** 2
            )

            style_loss = 0.0
            weight_sum = 0.0
            for layer,layer_weight in STYLE_LAYER_WEIGHTS.items():
                generated_gram = gram_matrix(generated_features[layer])
                target_gram = style_targets[layer]
                layer_style_loss = torch.mean((generated_gram - target_gram )**2)
                style_loss += layer_weight * layer_style_loss
                weight_sum += layer_weight

            style_loss = style_loss / weight_sum

            tv_loss = (
                torch.mean(torch.abs(generated[:,:,:,:-1] - generated[:,:,:,:1])) +
                torch.mean(torch.abs(generated[:,:,:-1,:] - generated[:,:,:1,:]))
            )

            total_loss = (
                CONTENT_WEIGHTS* content_loss
                + STYLE_WEIGHTS * style_loss
                + TV_WEIGHT * tv_loss
            )

            total_loss.backward()

            if step[0] % 50 == 0:
                print(
                    f"Step [{step[0]}/{NUM_STEPS}]"
                    f"Content Loss : {content_loss.item():.4f}"
                    f"Style Loss : {style_loss.item():.6e}"
                    f"Total Loss : {total_loss.item():.4f}"
                )
            step[0] += 1
            return total_loss
        

        optimizer.step(closure)

        with torch.no_grad():
            generated.clamp_(-3.0,3.0)
    save_tensor_image(generated, OUTPUT_IMAGE_PATH)
    print(f"Saved stylized image to: {OUTPUT_IMAGE_PATH}")

if __name__  == "__main__":
    run_style_transfer()

    