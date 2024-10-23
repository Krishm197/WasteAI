import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification,
    CLIPProcessor, 
    CLIPModel
)

class DualWasteClassifier:
    def __init__(self):
        # Initialize ResNet-50
        self.resnet_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
        self.resnet_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        
        # Initialize CLIP
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.resnet_model = self.resnet_model.to(self.device)
        self.clip_model = self.clip_model.to(self.device)
        
        # Categories for CLIP model
        self.clip_categories = [
            "recyclable plastic waste like plastic bottles and containers",
            "paper waste like newspapers and cardboard",
            "organic waste like food scraps and plant materials",
            "electronic waste like old phones and computers",
            "glass waste like bottles and jars",
            "metal waste like cans and foil",
            "hazardous waste like batteries and chemicals",
            "general non-recyclable waste"
        ]

    def get_resnet_prediction(self, image):
        # Process image for ResNet
        inputs = self.resnet_processor(image, return_tensors="pt").to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.resnet_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        
        # Get highest confidence prediction
        max_prob, max_idx = torch.max(probs, 0)
        category = self.resnet_model.config.id2label[max_idx.item()]
        confidence = max_prob.item() * 100
        
        return {
            'category': category,
            'confidence': round(confidence, 2)
        }

    def get_clip_prediction(self, image):
        # Process image and text with CLIP
        inputs = self.clip_processor(
            images=image,
            text=self.clip_categories,
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        outputs = self.clip_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits_per_image, dim=1)[0]
        
        # Get highest confidence prediction
        max_prob, max_idx = torch.max(probs, 0)
        category = self.clip_categories[max_idx.item()].split(' like ')[0]
        
        return {
            'category': category,
            'confidence': round(max_prob.item() * 100, 2)
        }

    def classify_image(self, image_path):
        # Load and convert image to RGB
        image = Image.open(image_path).convert('RGB')
        
        # Get predictions from both models
        resnet_result = self.get_resnet_prediction(image)
        clip_result = self.get_clip_prediction(image)
        
        # Format the combined result
        result = f"This is {resnet_result['category']} with {resnet_result['confidence']}% confidence "
        result += f"and the waste type is {clip_result['category']}"
        
        return result

def demo_classification():
    # Initialize classifier
    classifier = DualWasteClassifier()
    
    # Replace with your image path
    image_path = "waste_image.jpg"
    result = classifier.classify_image(image_path)
    print("\nClassification Result:")
    print(result)

# Example usage code
if __name__ == "__main__":
    demo_classification()
