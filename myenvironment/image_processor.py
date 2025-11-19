import numpy as np
import cv2
from PIL import Image
import pydicom
import torch
import torchvision.transforms as transforms
import time
import os

class ImageProcessor:
    def __init__(self):
        """Initialize image processor with transforms"""
        self.target_size = (224, 224)
        
        # Standard normalization for medical images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
    
    def validate_file(self, file_path, max_size_mb=10):
        """
        Validate uploaded file
        
        Args:
            file_path: Path to uploaded file
            max_size_mb: Maximum file size in MB
        
        Returns:
            tuple: (is_valid, error_message)
        """
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File too large: {file_size_mb:.2f}MB (max: {max_size_mb}MB)"
        
        # Check file extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.dcm', '.dicom']
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in valid_extensions:
            return False, f"Invalid file type: {file_ext}. Allowed: {valid_extensions}"
        
        # Basic check: try to load the image
        try:
            if file_ext in ['.dcm', '.dicom']:
                import pydicom
                pydicom.dcmread(file_path)
            else:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return False, "Unable to read image file. File may be corrupted."
                
                # Check if image is grayscale-like (medical images are typically grayscale)
                # This is a soft warning - we still allow color images
                if len(img.shape) == 3:
                    print("Note: Image appears to be color. Medical X-rays are typically grayscale.")
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
        
        return True, None
    
    def load_dicom(self, file_path):
        """
        Load and de-identify DICOM file
        
        Args:
            file_path: Path to DICOM file
        
        Returns:
            numpy array: Pixel data
        """
        # Read DICOM file
        dicom_data = pydicom.dcmread(file_path)
        
        # DE-IDENTIFY: Remove patient information (CRITICAL for privacy)
        tags_to_remove = [
            'PatientName', 'PatientID', 'PatientBirthDate',
            'PatientSex', 'PatientAge', 'InstitutionName',
            'ReferringPhysicianName', 'StudyDate', 'StudyTime'
        ]
        
        for tag in tags_to_remove:
            if hasattr(dicom_data, tag):
                setattr(dicom_data, tag, '')
        
        # Extract pixel data
        pixel_array = dicom_data.pixel_array
        
        # Normalize to 0-255 range
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
        pixel_array = pixel_array.astype(np.uint8)
        
        return pixel_array
    
    def load_image(self, file_path):
        """
        Load image from various formats
        
        Args:
            file_path: Path to image file
        
        Returns:
            numpy array: Grayscale image
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.dcm', '.dicom']:
            # Load DICOM
            image = self.load_dicom(file_path)
        else:
            # Load regular image
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                # Try with PIL as backup
                pil_image = Image.open(file_path).convert('L')
                image = np.array(pil_image)
        
        return image
    
    def preprocess(self, file_path):
        """
        Complete preprocessing pipeline
        
        Args:
            file_path: Path to uploaded file
        
        Returns:
            tuple: (tensor, preprocessing_time_ms, original_image)
        """
        start_time = time.time()
        
        # Load image
        image = self.load_image(file_path)
        
        # Store original for visualization
        original_image = image.copy()
        
        # Resize to target size
        image_resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Convert to PIL Image for transforms
        pil_image = Image.fromarray(image_resized)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        # Add batch dimension [1, 1, 224, 224]
        tensor = tensor.unsqueeze(0)
        
        preprocess_time = (time.time() - start_time) * 1000
        
        return tensor, preprocess_time, original_image

# Global processor instance
image_processor = ImageProcessor()