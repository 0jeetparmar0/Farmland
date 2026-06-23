import os
import torch
import rasterio
from samgeo import SamGeo
import numpy as np
import cv2
from typing import Optional, List
from tqdm import tqdm
import gc

class SAMBinaryMaskGenerator:
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam = self._initialize_sam(checkpoint_path)
        
    def _initialize_sam(self, checkpoint_path: str) -> SamGeo:
        """Initialize SAM model with optimized settings"""
        return SamGeo(
            checkpoint=checkpoint_path,
            model_type='vit_h',
            device=self.device,
            erosion_kernel=(3, 3),
            mask_multiplier=255,
            sam_kwargs={
                'points_per_side': 32,
                'pred_iou_thresh': 0.86,
                'stability_score_thresh': 0.92,
                'crop_n_layers': 1,
                'crop_n_points_downscale_factor': 2,
                'min_mask_region_area': 100,
            },
        )
    
    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Enhance mask quality with morphological operations"""
        # Remove small artifacts
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fill small holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Ensure binary output
        _, binary = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
        return binary

    def process_directory(
        self,
        input_directory: str,
        output_directory: str,
        file_extensions: List[str] = ['.tif', '.tiff'],
        max_files: Optional[int] = None,
        skip_existing: bool = True
    ) -> None:
        """Process all images in directory to create optimized binary masks"""
        os.makedirs(os.path.join(output_directory, "masks"), exist_ok=True)
        
        files = [
            f for f in os.listdir(input_directory) 
            if any(f.lower().endswith(ext) for ext in file_extensions)
        ]
        
        if max_files:
            files = files[:max_files]
            
        for filename in tqdm(files, desc="Generating Masks"):
            try:
                base_name = os.path.splitext(filename)[0]
                mask_path = os.path.join(output_directory, "masks", f"{base_name}_mask.tif")
                
                if skip_existing and os.path.exists(mask_path):
                    continue
                    
                self._process_single_file(
                    input_path=os.path.join(input_directory, filename),
                    output_path=mask_path
                )
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
            finally:
                gc.collect()
                torch.cuda.empty_cache()

    def _process_single_file(self, input_path: str, output_path: str) -> None:
        """Process a single image file"""
        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            profile.update(
                count=1,
                dtype='uint8',
                compress='lzw',  # Lossless compression
                nodata=0
            )
            
            # Generate mask directly to memory
            temp_mask = "temp_mask.tif"
            self.sam.generate(input_path, temp_mask)
            
            with rasterio.open(temp_mask) as mask_src:
                masks = mask_src.read()
                combined = np.any(masks, axis=0).astype(np.uint8) * 255
                processed = self._postprocess_mask(combined)
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(processed, 1)
            
            if os.path.exists(temp_mask):
                os.remove(temp_mask)

if __name__ == "__main__":
    # Configuration
    config = {
        "input_dir": "/home/jazzy/sem-farm/data",
        "output_dir": "/home/jazzy/sem-farm/SEGMENTATION/SamGeo/output",
        "checkpoint": "/home/jazzy/sem-farm/SEGMENTATION/SamGeo/model/sam_vit_h_4b8939.pth",
        "max_files": None,  # Set to number to limit processing
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    # Initialize processor
    processor = SAMBinaryMaskGenerator(
        checkpoint_path=config["checkpoint"],
        device=config["device"]
    )
    
    # Run processing
    processor.process_directory(
        input_directory=config["input_dir"],
        output_directory=config["output_dir"],
        max_files=config["max_files"]
    )