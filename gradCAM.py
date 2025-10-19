import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

class TFLiteGradCAM:
    def __init__(self, model_path, layer_name=None):

        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape and dtype
        self.input_shape = self.input_details[0]['shape'][1:3]
        self.input_dtype = self.input_details[0]['dtype']
        
        print(f"Model expects input dtype: {self.input_dtype}")
        print(f"Model input shape: {self.input_details[0]['shape']}")
        
    def preprocess_image(self, img_path):
        """Load and preprocess image"""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, tuple(self.input_shape))
        img_array = np.expand_dims(img_resized, axis=0)
        
        # Convert to the expected dtype
        if self.input_dtype == np.uint8:
            img_array = img_array.astype(np.uint8)  # Keep in 0-255 range
        else:
            img_array = img_array.astype(np.float32) / 255.0  # Normalize to [0,1]
        
        return img, img_array
    
    def get_gradients(self, img_array, class_idx=None):
        """
        Approximate gradients using finite differences
        This is a workaround since TFLite doesn't support gradients
        """
        # Get base prediction
        self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
        self.interpreter.invoke()
        base_output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        if class_idx is None:
            class_idx = np.argmax(base_output)
        
        # Calculate gradients using finite differences
        epsilon = 1 if self.input_dtype == np.uint8 else 1e-3
        gradients = np.zeros_like(img_array, dtype=np.float32)
        
        for i in range(img_array.shape[1]):
            for j in range(img_array.shape[2]):
                for k in range(img_array.shape[3]):
                    # Perturb pixel
                    perturbed = img_array.copy()
                    
                    # Ensure we don't exceed bounds
                    if self.input_dtype == np.uint8:
                        new_val = min(255, int(perturbed[0, i, j, k]) + int(epsilon))
                        perturbed[0, i, j, k] = new_val
                    else:
                        perturbed[0, i, j, k] = min(1.0, perturbed[0, i, j, k] + epsilon)
                    
                    # Get prediction
                    self.interpreter.set_tensor(self.input_details[0]['index'], perturbed)
                    self.interpreter.invoke()
                    perturbed_output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                    
                    # Approximate gradient
                    gradients[0, i, j, k] = (perturbed_output[class_idx] - base_output[class_idx]) / epsilon
        
        return gradients, class_idx, base_output
    
    def generate_heatmap(self, img_array, class_idx=None):
        """Generate GradCAM heatmap"""
        gradients, pred_class, output = self.get_gradients(img_array, class_idx)
        
        # Convert img_array to float for processing
        img_float = img_array.astype(np.float32)
        if self.input_dtype == np.uint8:
            img_float = img_float / 255.0
        
        # Pool gradients across channels
        weights = np.mean(gradients[0], axis=(0, 1))
        
        # Weighted combination
        cam = np.zeros(img_array.shape[1:3], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * img_float[0, :, :, i]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam, pred_class, output
    
    def overlay_heatmap(self, img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """Overlay heatmap on original image"""
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert to uint8
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlayed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlayed


def process_four_images(model_path, image_paths, class_names=None):

    if len(image_paths) != 4:
        raise ValueError("Please provide exactly 4 image paths")
    
    # Initialize GradCAM
    gradcam = TFLiteGradCAM(model_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('GradCAM Visualization on 4 Images', fontsize=16, fontweight='bold')
    
    for idx, img_path in enumerate(image_paths):
        print(f"\nProcessing image {idx+1}/4: {img_path}")
        row = idx // 2
        col_orig = (idx % 2) * 2
        col_grad = col_orig + 1
        
        # Process image
        original_img, preprocessed_img = gradcam.preprocess_image(img_path)
        
        # Generate heatmap
        print(f"Generating GradCAM heatmap (this may take a while)...")
        heatmap, pred_class, output = gradcam.generate_heatmap(preprocessed_img)
        
        # Overlay heatmap
        overlayed = gradcam.overlay_heatmap(original_img, heatmap)
        
        # Get prediction info
        confidence = output[pred_class] * 100
        if class_names and pred_class < len(class_names):
            pred_label = class_names[pred_class]
        else:
            pred_label = f"Class {pred_class}"
        
        print(f"Prediction: {pred_label} ({confidence:.1f}%)")
        
        # Display original image
        axes[row, col_orig].imshow(original_img)
        axes[row, col_orig].set_title(f'Image {idx+1}\n{pred_label} ({confidence:.1f}%)', 
                                       fontsize=10, fontweight='bold')
        axes[row, col_orig].axis('off')
        
        # Display GradCAM
        axes[row, col_grad].imshow(overlayed)
        axes[row, col_grad].set_title(f'GradCAM {idx+1}', fontsize=10, fontweight='bold')
        axes[row, col_grad].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*50)
    print("GradCAM visualization completed!")
    print("Results saved to 'gradcam_results.png'")
    print("="*50)


# Example usage
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "exports/model_int8.tflite"  # Your TFLite model path
    IMAGE_PATHS = [
        "Tire gradcam/test_images/Break-in Period/IMG_7664.MOV-1.jpg",
        "Tire gradcam/test_images/Change Soon/IMG_7668.MOV-1.jpg", 
        "Tire gradcam/test_images/Good/IMG_7672.JPG",
        "Tire gradcam/test_images/Unsafe to Drive/IMG_7669.MOV-1.jpg"
    ]
    
    # Optional: Define class names
    CLASS_NAMES = ['Good', 'Change Soon', 'Break-in Period', 'Unsafe to Drive']
    
    # Run GradCAM on 4 images
    process_four_images(MODEL_PATH, IMAGE_PATHS, CLASS_NAMES)
