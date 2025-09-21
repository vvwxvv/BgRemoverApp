import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SmartBackgroundRemover:
    """
    Production-ready white background remover with intelligent edge detection
    and object preservation.
    """
    
    def __init__(self):
        """Initialize the background remover with optimized default settings."""
        self.min_object_ratio = 0.001  # Minimum object size as ratio of image
        self.edge_refinement_iterations = 2
        self.smoothing_kernel_size = 3
        
    def remove_background(self, 
                         image_path: Union[str, Path], 
                         output_path: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main method to remove white/light background from an image.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save the result
            
        Returns:
            Tuple of (result_image_rgba, mask)
        """
        # Load and validate image
        image = self._load_image(image_path)
        
        # Intelligently detect and create mask
        mask = self._create_intelligent_mask(image)
        
        # Refine mask edges for better quality
        mask = self._refine_mask_edges(image, mask)
        
        # Create RGBA result
        result = self._apply_mask_to_image(image, mask)
        
        # Save if output path provided
        if output_path:
            self._save_result(result, output_path)
        
        return result, mask
    
    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load and validate image."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        logger.info(f"Loaded image: {image_path} ({image.shape[1]}x{image.shape[0]})")
        return image
    
    def _create_intelligent_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create an intelligent mask that preserves objects while removing white background.
        Uses multiple detection strategies and combines them optimally.
        """
        height, width = image.shape[:2]
        
        # Strategy 1: White color detection with adaptive threshold
        white_mask = self._detect_white_regions(image)
        
        # Strategy 2: Edge-based object detection
        object_mask = self._detect_objects_by_edges(image)
        
        # Strategy 3: Saturation-based detection (colored objects)
        color_mask = self._detect_colored_regions(image)
        
        # Combine strategies intelligently
        # Start with inverted white mask (non-white areas are objects)
        combined_mask = cv2.bitwise_not(white_mask)
        
        # Include detected objects
        combined_mask = cv2.bitwise_or(combined_mask, object_mask)
        
        # Include colored regions
        combined_mask = cv2.bitwise_or(combined_mask, color_mask)
        
        # Remove small noise
        combined_mask = self._remove_small_components(combined_mask, height * width)
        
        # Fill holes in objects
        combined_mask = self._fill_object_holes(combined_mask)
        
        return combined_mask
    
    def _detect_white_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Detect white/light background regions with adaptive thresholding.
        """
        # Convert to LAB color space for better white detection
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate adaptive threshold based on image statistics
        mean_brightness = np.mean(l_channel)
        std_brightness = np.std(l_channel)
        
        # Dynamic threshold: typically white backgrounds are in top 10-15% brightness
        threshold = min(mean_brightness + 1.5 * std_brightness, 240)
        threshold = max(threshold, 200)  # Ensure minimum threshold for white detection
        
        # Create white mask
        _, white_mask = cv2.threshold(l_channel, threshold, 255, cv2.THRESH_BINARY)
        
        # Check if edges are predominantly white (indicates white background)
        edge_pixels = self._get_edge_pixels(l_channel)
        edge_mean = np.mean(edge_pixels)
        
        if edge_mean > 200:  # Edges are very bright, likely white background
            # Relax threshold slightly to catch more background
            _, white_mask = cv2.threshold(l_channel, threshold - 10, 255, cv2.THRESH_BINARY)
        
        return white_mask
    
    def _detect_objects_by_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect objects using intelligent edge detection.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Calculate dynamic Canny thresholds based on image statistics
        median_val = np.median(filtered)
        lower = int(max(0, 0.66 * median_val))
        upper = int(min(255, 1.33 * median_val))
        
        # Detect edges
        edges = cv2.Canny(filtered, lower, upper)
        
        # Connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find and fill contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        min_area = self.min_object_ratio * gray.shape[0] * gray.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.fillPoly(mask, [contour], 255)
        
        return mask
    
    def _detect_colored_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Detect colored (non-white/gray) regions using HSV color space.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect regions with significant saturation (colored areas)
        saturation = hsv[:, :, 1]
        
        # Adaptive threshold for saturation
        mean_sat = np.mean(saturation)
        threshold = max(20, mean_sat * 0.5)  # Dynamic threshold
        
        _, color_mask = cv2.threshold(saturation, threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        return color_mask
    
    def _get_edge_pixels(self, image: np.ndarray) -> np.ndarray:
        """Get pixels from image borders for background analysis."""
        h, w = image.shape[:2] if len(image.shape) == 2 else image.shape[:2]
        border_width = max(5, min(h, w) // 100)  # Adaptive border width
        
        edge_pixels = []
        edge_pixels.extend(image[:border_width, :].flatten())  # Top
        edge_pixels.extend(image[-border_width:, :].flatten())  # Bottom
        edge_pixels.extend(image[:, :border_width].flatten())  # Left
        edge_pixels.extend(image[:, -border_width:].flatten())  # Right
        
        return np.array(edge_pixels)
    
    def _remove_small_components(self, mask: np.ndarray, image_area: int) -> np.ndarray:
        """Remove small noise components from mask."""
        # Find all connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Calculate minimum component size (adaptive)
        min_size = max(100, int(image_area * self.min_object_ratio))
        
        # Create new mask keeping only large components
        result_mask = np.zeros_like(mask)
        for label in range(1, num_labels):  # Skip background (label 0)
            if stats[label, cv2.CC_STAT_AREA] >= min_size:
                result_mask[labels == label] = 255
        
        return result_mask
    
    def _fill_object_holes(self, mask: np.ndarray) -> np.ndarray:
        """Fill holes inside detected objects."""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Fill contours (fills holes)
        filled_mask = np.zeros_like(mask)
        cv2.fillPoly(filled_mask, contours, 255)
        
        return filled_mask
    
    def _refine_mask_edges(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Refine mask edges using GrabCut-inspired technique for better quality.
        """
        # Create trimap from mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Erode to get definite foreground
        fg_mask = cv2.erode(mask, kernel, iterations=self.edge_refinement_iterations)
        
        # Dilate to get possible foreground/background boundary
        boundary_mask = cv2.dilate(mask, kernel, iterations=self.edge_refinement_iterations)
        
        # Create trimap
        trimap = np.zeros_like(mask)
        trimap[boundary_mask == 255] = 128  # Possible foreground/background
        trimap[fg_mask == 255] = 255  # Definite foreground
        
        # Refine edges using color similarity
        refined_mask = self._refine_by_color_similarity(image, mask, trimap)
        
        # Smooth the final mask
        refined_mask = cv2.medianBlur(refined_mask, self.smoothing_kernel_size)
        
        return refined_mask
    
    def _refine_by_color_similarity(self, image: np.ndarray, mask: np.ndarray, 
                                   trimap: np.ndarray) -> np.ndarray:
        """
        Refine uncertain pixels based on color similarity to foreground/background.
        """
        refined_mask = mask.copy()
        
        # Get uncertain pixels
        uncertain_pixels = (trimap == 128)
        
        if not np.any(uncertain_pixels):
            return refined_mask
        
        # Convert to LAB for better color comparison
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Get definite foreground and background pixels
        fg_pixels = lab[mask == 255]
        bg_pixels = lab[mask == 0]
        
        if len(fg_pixels) == 0 or len(bg_pixels) == 0:
            return refined_mask
        
        # Calculate mean colors
        fg_mean = np.mean(fg_pixels, axis=0)
        bg_mean = np.mean(bg_pixels, axis=0)
        
        # Classify uncertain pixels
        uncertain_coords = np.where(uncertain_pixels)
        for y, x in zip(uncertain_coords[0], uncertain_coords[1]):
            pixel = lab[y, x]
            
            # Calculate distances
            fg_dist = np.linalg.norm(pixel - fg_mean)
            bg_dist = np.linalg.norm(pixel - bg_mean)
            
            # Classify based on closer mean
            refined_mask[y, x] = 255 if fg_dist < bg_dist else 0
        
        return refined_mask
    
    def _apply_mask_to_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to create RGBA image with transparent background."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create RGBA image
        height, width = image_rgb.shape[:2]
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Copy RGB channels
        rgba[:, :, :3] = image_rgb
        
        # Set alpha channel
        rgba[:, :, 3] = mask
        
        return rgba
    
    def _save_result(self, result: np.ndarray, output_path: Union[str, Path]) -> None:
        """Save the result as PNG with transparency."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        pil_image = Image.fromarray(result, 'RGBA')
        pil_image.save(str(output_path), 'PNG', optimize=True)
        logger.info(f"Result saved to: {output_path}")
    
    def process_batch(self, input_folder: Union[str, Path], 
                     output_folder: Union[str, Path]) -> None:
        """
        Process multiple images in a folder.
        
        Args:
            input_folder: Folder containing input images
            output_folder: Folder to save processed images
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in input_folder.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        logger.info(f"Processing {len(image_files)} images...")
        
        for image_file in image_files:
            try:
                output_file = output_folder / f"{image_file.stem}_no_bg.png"
                self.remove_background(image_file, output_file)
                logger.info(f"Processed: {image_file.name}")
            except Exception as e:
                logger.error(f"Failed to process {image_file.name}: {e}")
        
        logger.info("Batch processing completed!")


def visualize_results(original_path: Union[str, Path], 
                      result: np.ndarray, 
                      mask: np.ndarray) -> None:
    """
    Visualize the original image, mask, and result.
    
    Args:
        original_path: Path to original image
        result: RGBA result image
        mask: Binary mask
    """
    # Load original
    original = cv2.imread(str(original_path))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Generated Mask', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Result with transparency
    axes[2].imshow(result)
    axes[2].set_title('Transparent Background', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Result on colored background (to show transparency)
    colored_bg = np.full_like(result, [135, 206, 235, 255])  # Sky blue
    composite = np.zeros_like(result)
    alpha = result[:, :, 3:4] / 255.0
    composite[:, :, :3] = (result[:, :, :3] * alpha + 
                           colored_bg[:, :, :3] * (1 - alpha)).astype(np.uint8)
    composite[:, :, 3] = 255
    
    axes[3].imshow(composite)
    axes[3].set_title('Result on Colored Background', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    plt.suptitle('Smart Background Removal Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize the remover
    remover = SmartBackgroundRemover()
    
    # Single image processing
    try:
        input_image = "1.png"  # Your input image
        output_image = "output_no_background.png"
        
        # Process image
        result, mask = remover.remove_background(input_image, output_image)
        
        # Visualize results
        visualize_results(input_image, result, mask)
        
        print(f"\n✅ Successfully processed image!")
        print(f"📁 Output saved to: {output_image}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure the input image exists.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your image format and try again.")
    
