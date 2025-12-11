"""
Unified Damage Detection Pipeline
Combines Parts Segmentation + Damage Detection + Severity Classification
"""
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger

# Try timm first (for training compatibility), fallback to torchvision
try:
    import timm
    USE_TIMM = True
    logger.info("Using timm library for EfficientNet")
except ImportError:
    from torchvision.models import efficientnet_b0
    USE_TIMM = False
    logger.warning("timm not found, using torchvision (may have compatibility issues)")


class DamageDetector:
    """
    End-to-end damage detection pipeline:
    1. Detect car parts (YOLO11n-seg) - 23 classes
    2. Detect damage on parts (YOLO11m) - 6 damage types
    3. Classify damage severity (EfficientNet-B0) - 3 levels
    """
    
    # Class names from training
    PARTS_CLASSES = [
        'backbumper', 'backdoor', 'backglass', 'backleftdoor', 'backleftlight',
        'backlight', 'backrightdoor', 'backrightlight', 'frontbumper', 'frontdoor',
        'frontglass', 'frontleftdoor', 'frontleftlight', 'frontlight',
        'frontrightdoor', 'frontrightlight', 'hood', 'leftmirror', 'roof',
        'rightmirror', 'tailgate', 'trunk', 'wheel'
    ]
    
    DAMAGE_CLASSES = [
        'dent', 'scratch', 'crack', 'glass-shatter', 'tire-flat', 'lamp-broken'
    ]
    
    SEVERITY_CLASSES = ['minor', 'moderate', 'severe']
    
    def __init__(
        self,
        parts_model_path: str,
        damage_model_path: str,
        severity_model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize damage detector with all three models
        
        Args:
            parts_model_path: Path to YOLO11n-seg parts model
            damage_model_path: Path to YOLO11m damage model
            severity_model_path: Path to EfficientNet-B0 severity model
            device: Device to run inference on (cuda/cpu)
        """
        self.device = device
        logger.info(f"Initializing DamageDetector on {device}")
        
        # Load YOLO models
        logger.info("Loading parts segmentation model...")
        self.parts_model = YOLO(parts_model_path)
        self.parts_model.to(device)
        
        logger.info("Loading damage detection model...")
        self.damage_model = YOLO(damage_model_path)
        self.damage_model.to(device)
        
        # Load severity classifier
        logger.info("Loading severity classification model...")
        self.severity_model = self._load_severity_model(severity_model_path)
        self.severity_model.to(device)
        self.severity_model.eval()
        
        # Severity preprocessing transforms
        self.severity_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.success("✓ All models loaded successfully")
    
    def _load_severity_model(self, model_path: str) -> nn.Module:
        """
        Load EfficientNet-B0 severity classifier
        
        Args:
            model_path: Path to .pth checkpoint
            
        Returns:
            Loaded PyTorch model
        """
        if USE_TIMM:
            # Load using timm (matches training)
            model = timm.create_model(
                'efficientnet_b0',
                pretrained=False,
                num_classes=len(self.SEVERITY_CLASSES)
            )
        else:
            # Fallback to torchvision
            from torchvision.models import efficientnet_b0
            model = efficientnet_b0(pretrained=False)
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features,
                len(self.SEVERITY_CLASSES)
            )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                # Assume the dict itself is the state_dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load with strict=False to handle minor mismatches
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info("Loaded severity model with strict=True")
        except Exception as e:
            logger.warning(f"Strict loading failed, trying strict=False: {str(e)[:100]}")
            model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded severity model with strict=False (some weights may not match)")
        
        return model
    
    def detect_damage(
        self,
        image_path: str,
        parts_conf: float = 0.25,
        damage_conf: float = 0.25,
        return_visualization: bool = False
    ) -> Dict:
        """
        Complete damage detection pipeline
        
        Args:
            image_path: Path to input image
            parts_conf: Confidence threshold for parts detection
            damage_conf: Confidence threshold for damage detection
            return_visualization: Whether to return annotated image
        
        Returns:
            Dictionary with detection results:
            {
                "parts_detected": [...],
                "damages_detected": [...],
                "summary": {...},
                "risk_assessment": {...}
            }
        """
        logger.info(f"Processing image: {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        results = {
            "parts_detected": [],
            "damages_detected": [],
            "summary": {},
            "risk_assessment": {}
        }
        
        # Step 1: Detect car parts
        logger.info("Step 1: Detecting car parts...")
        parts_results = self.parts_model.predict(
            image_path,
            conf=parts_conf,
            verbose=False
        )[0]
        
        parts_detected = []
        for box in parts_results.boxes:
            parts_detected.append({
                "class": parts_results.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].cpu().numpy().tolist()
            })
        
        results["parts_detected"] = parts_detected
        logger.info(f"  Found {len(parts_detected)} car parts")
        
        # Step 2: Detect damage
        logger.info("Step 2: Detecting damage...")
        damage_results = self.damage_model.predict(
            image_path,
            conf=damage_conf,
            verbose=False
        )[0]
        
        damages_detected = []
        for box in damage_results.boxes:
            damage_class = damage_results.names[int(box.cls)]
            bbox = box.xyxy[0].cpu().numpy().tolist()
            
            # Crop damage region for severity classification
            x1, y1, x2, y2 = map(int, bbox)
            damage_crop = image[y1:y2, x1:x2]
            
            # Step 3: Classify severity for this damage
            severity = self._classify_severity(damage_crop)
            
            damages_detected.append({
                "damage_type": damage_class,
                "confidence": float(box.conf),
                "bbox": bbox,
                "severity": severity["class"],
                "severity_confidence": severity["confidence"]
            })
        
        results["damages_detected"] = damages_detected
        logger.info(f"  Found {len(damages_detected)} damages")
        
        # Generate summary
        results["summary"] = self._generate_summary(damages_detected)
        results["risk_assessment"] = self._assess_risk(damages_detected, parts_detected)
        
        # Optional: Create visualization
        if return_visualization:
            results["visualization"] = self._create_visualization(
                image, parts_detected, damages_detected
            )
        
        logger.success("✓ Damage detection complete")
        return results
    
    def _classify_severity(self, crop_image: np.ndarray) -> Dict:
        """
        Classify damage severity for a cropped damage region
        
        Args:
            crop_image: BGR image crop (numpy array)
            
        Returns:
            Dict with class name and confidence
        """
        if crop_image.size == 0 or crop_image.shape[0] < 10 or crop_image.shape[1] < 10:
            return {"class": "unknown", "confidence": 0.0}
        
        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)
        
        # Apply transforms
        input_tensor = self.severity_transform(pil_image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.severity_model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        severity_class = self.SEVERITY_CLASSES[predicted.item()]
        confidence_score = confidence.item()
        
        return {
            "class": severity_class,
            "confidence": round(confidence_score, 3)
        }
    
    def _generate_summary(self, damages: List[Dict]) -> Dict:
        """
        Generate summary statistics from detected damages
        
        Args:
            damages: List of damage detections
            
        Returns:
            Summary dictionary
        """
        if not damages:
            return {
                "total_damages": 0,
                "damage_types": {},
                "severity_distribution": {},
                "most_severe": None
            }
        
        # Count damage types
        damage_types = {}
        for damage in damages:
            dtype = damage["damage_type"]
            damage_types[dtype] = damage_types.get(dtype, 0) + 1
        
        # Count severity levels
        severity_dist = {}
        for damage in damages:
            severity = damage["severity"]
            severity_dist[severity] = severity_dist.get(severity, 0) + 1
        
        # Find most severe damage
        severity_order = {"severe": 3, "moderate": 2, "minor": 1, "unknown": 0}
        most_severe = max(
            damages,
            key=lambda d: severity_order.get(d["severity"], 0)
        )
        
        return {
            "total_damages": len(damages),
            "damage_types": damage_types,
            "severity_distribution": severity_dist,
            "most_severe_damage": {
                "type": most_severe["damage_type"],
                "severity": most_severe["severity"]
            }
        }
    
    def _assess_risk(self, damages: List[Dict], parts: List[Dict]) -> Dict:
        """
        Assess claim risk based on detected damages
        
        Args:
            damages: List of damage detections
            parts: List of part detections
            
        Returns:
            Risk assessment dictionary
        """
        if not damages:
            return {
                "risk_level": "LOW",
                "risk_score": 0.0,
                "factors": []
            }
        
        risk_score = 0.0
        factors = []
        
        # Severity-based risk
        severe_count = sum(1 for d in damages if d["severity"] == "severe")
        moderate_count = sum(1 for d in damages if d["severity"] == "moderate")
        
        if severe_count > 0:
            risk_score += severe_count * 0.3
            factors.append(f"{severe_count} severe damage(s)")
        
        if moderate_count > 0:
            risk_score += moderate_count * 0.15
            factors.append(f"{moderate_count} moderate damage(s)")
        
        # Multiple damage types risk
        unique_damages = len(set(d["damage_type"] for d in damages))
        if unique_damages >= 3:
            risk_score += 0.2
            factors.append(f"Multiple damage types ({unique_damages})")
        
        # Critical parts damaged (glass, structural)
        critical_parts = ["frontglass", "backglass", "hood", "roof"]
        damaged_critical = any(
            any(part["class"] in critical_parts for part in parts)
            for _ in damages  # simplified check
        )
        if damaged_critical:
            risk_score += 0.15
            factors.append("Critical parts may be affected")
        
        # Normalize score to 0-1
        risk_score = min(risk_score, 1.0)
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "HIGH"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "risk_level": risk_level,
            "risk_score": round(risk_score, 3),
            "factors": factors
        }
    
    def _create_visualization(
        self,
        image: np.ndarray,
        parts: List[Dict],
        damages: List[Dict]
    ) -> str:
        """
        Create annotated visualization (placeholder)
        
        Args:
            image: Original image
            parts: Part detections
            damages: Damage detections
            
        Returns:
            Base64 encoded image string (to be implemented)
        """
        # TODO: Implement visualization with cv2.rectangle and labels
        return "visualization_placeholder"
