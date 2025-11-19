import torch
import torchxrayvision as xrv
import numpy as np
import time

class XRayModelHandler:
    def __init__(self):
        """Initialize and load the pretrained model"""
        print("Loading pretrained model...")
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.model.eval()  # Set to evaluation mode
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Store pathology labels
        self.pathologies = self.model.pathologies
        print(f"Model loaded successfully on {self.device}")
        print(f"Pathologies: {self.pathologies}")
    
    def predict(self, image_tensor):
        """
        Make prediction on preprocessed image
        
        Args:
            image_tensor: Preprocessed image tensor [1, 1, 224, 224]
        
        Returns:
            dict: Contains predictions, probabilities, and timing
        """
        start_time = time.time()
        
        # Move tensor to device
        image_tensor = image_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        inference_time = time.time() - start_time
        
        # Convert to probabilities using sigmoid
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Create results dictionary
        results = {
            pathology: float(prob) 
            for pathology, prob in zip(self.pathologies, probabilities)
        }
        
        # Sort by probability (highest first)
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'predictions': results,
            'inference_time_ms': inference_time * 1000,
            'device': str(self.device)
        }
    
    def get_feature_maps(self, image_tensor):
        """
        Extract feature maps for Grad-CAM
        
        Returns:
            features and model output for explainability
        """
        image_tensor = image_tensor.to(self.device)
        
        # Hook to capture feature maps
        features = []
        def hook_fn(module, input, output):
            features.append(output)
        
        # Register hook on the last convolutional layer
        # For DenseNet, this is typically the last dense block
        target_layer = self.model.features[-1]
        handle = target_layer.register_forward_hook(hook_fn)
        
        # Forward pass
        output = self.model(image_tensor)
        
        # Remove hook
        handle.remove()
        
        return features[0], output

# Global model instance (loaded once at app startup)
model_handler = None

def initialize_model():
    """Initialize model at app startup"""
    global model_handler
    if model_handler is None:
        model_handler = XRayModelHandler()
    return model_handler