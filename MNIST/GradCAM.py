import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()
        self.device = device

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        layer = dict([*self.model.named_modules()])[self.target_layer]
        layer.register_forward_hook(forward_hook)
        layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class):
        # Forward pass
        output = self.model(input_tensor)
        # Zero grads and backward for the target class
        self.model.zero_grad()
        target = output[0][target_class]
        target.backward()

        # Get pooled gradients and activations

        gradients = self.gradients.mean(dim=[0, 2, 3])
        activations = self.activations[0]
        weights = gradients
        # Generate weighted combination of channels
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        cam = cam.to(self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            # print(cam)

        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        # Resize CAM to match input image size using bilinear interpolation: (32, 32)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.cpu()
        return cam.detach().numpy()


class GradCAMVisualizer:
    @staticmethod
    def preprocess_image(image_path):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)

    @staticmethod
    def show_cam_on_image(img_path, cam):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = np.float32(heatmap) / 255 + np.float32(img) / 255
        overlay = overlay / np.max(overlay)
        plt.imshow(np.uint8(255 * overlay))
        plt.axis('off')
        plt.show()


__all__ = ['GradCAM', 'GradCAMVisualizer']
