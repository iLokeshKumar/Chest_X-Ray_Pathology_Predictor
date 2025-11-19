import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained model
        """
        self.model = model
        self.feature_maps = None
        self.gradients = None
    
    def generate_cam(self, image_tensor, target_class_idx=None):
        """
        Generate Grad-CAM heatmap using a simplified approach
        
        Args:
            image_tensor: Input image tensor
            target_class_idx: Index of target class (None = highest prediction)
        
        Returns:
            numpy array: Heatmap overlay
        """
        # Store feature maps and gradients
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output.detach())
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())
        
        # Register hooks on the last conv layer
        target_layer = self.model.features[-1]
        fh = target_layer.register_forward_hook(forward_hook)
        bh = target_layer.register_backward_hook(backward_hook)
        
        # Enable gradients temporarily
        with torch.set_grad_enabled(True):
            # Create a fresh tensor copy
            img = image_tensor.detach().clone()
            img.requires_grad_(True)
            
            # Forward pass
            self.model.eval()
            output = self.model(img)
            
            # Get target class
            if target_class_idx is None:
                probabilities = torch.sigmoid(output)
                target_class_idx = torch.argmax(probabilities).item()
            
            # Zero gradients
            self.model.zero_grad()
            if img.grad is not None:
                img.grad.zero_()
            
            # Backward pass
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class_idx] = 1
            output.backward(gradient=one_hot, retain_graph=False)
        
        # Remove hooks
        fh.remove()
        bh.remove()
        
        # Process activations and gradients
        act = activations[0].cpu().numpy()[0]  # [C, H, W]
        grad = gradients[0].cpu().numpy()[0]   # [C, H, W]
        
        # Global average pooling on gradients
        weights = np.mean(grad, axis=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = np.zeros(act.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * act[i]
        
        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam, target_class_idx
    
    def create_overlay(self, original_image, cam, alpha=0.4):
        """
        Create heatmap overlay on original image
        
        Args:
            original_image: Original grayscale image
            cam: Grad-CAM heatmap
            alpha: Transparency of overlay
        
        Returns:
            numpy array: RGB image with heatmap overlay
        """
        # Resize CAM to match original image
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Convert to heatmap (0-255)
        heatmap = np.uint8(255 * cam_resized)
        
        # Apply colormap (red for high activation)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert original image to RGB
        if len(original_image.shape) == 2:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        else:
            original_rgb = original_image
        
        # Resize original to match if needed
        if original_rgb.shape[:2] != heatmap_colored.shape[:2]:
            original_rgb = cv2.resize(original_rgb, (heatmap_colored.shape[1], heatmap_colored.shape[0]))
        
        # Blend
        overlay = cv2.addWeighted(original_rgb, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay

def generate_explanation(model, image_tensor, original_image, top_k=3):
    """
    Generate explanations for top predictions
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        original_image: Original image for overlay
        top_k: Number of top predictions to explain
    
    Returns:
        dict: Explanations with heatmap paths
    """
    gradcam = GradCAM(model)
    
    # Get predictions
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.sigmoid(output)
    
    # Get top k predictions
    top_probs, top_indices = torch.topk(probabilities[0], top_k)
    
    explanations = []
    
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        # Generate CAM for this class
        cam, _ = gradcam.generate_cam(image_tensor, target_class_idx=idx.item())
        
        # Create overlay
        overlay = gradcam.create_overlay(original_image, cam)
        
        explanations.append({
            'class_name': model.pathologies[idx],
            'probability': float(prob),
            'heatmap': overlay,
            'rank': i + 1
        })
    
    return explanations