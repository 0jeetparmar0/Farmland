import os
import torch
import rasterio
from rasterio.windows import Window
from samgeo import SamGeo
import numpy as np
import cv2
from skimage.morphology import skeletonize
from tqdm import tqdm
import imageio


def process_tif_directory_chunked(input_directory, output_directory, checkpoint_path, no_of_files=1, chunk_size=512):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = SamGeo(
        checkpoint=checkpoint_path,
        model_type='vit_h',
        device=device,
        erosion_kernel=(3, 3),
        mask_multiplier=255,
        sam_kwargs=None,
    )

    tif_files = [f for f in os.listdir(input_directory) if f.endswith(".tif")][:no_of_files]

    for filename in tqdm(tif_files, desc="Processing files", unit="file"):
        tif_path = os.path.join(input_directory, filename)
        output_mask_path = os.path.join(output_directory, f"binary_mask_{filename}")
        output_vector = os.path.join(output_directory, f"vector_{filename[:-4]}.gpkg")

        with rasterio.open(tif_path) as src:
            profile = src.profile
            profile.update(dtype=rasterio.uint8, count=1)
            width, height = src.width, src.height

            windows = [
                Window(col_off, row_off, min(chunk_size, width - col_off), min(chunk_size, height - row_off))
                for row_off in range(0, height, chunk_size)
                for col_off in range(0, width, chunk_size)
            ]

            final_mask = np.zeros((height, width), dtype=np.uint8)

            for win in tqdm(windows, desc=f"Chunks of {filename}", unit="chunk", leave=False):
                left, top = int(win.col_off), int(win.row_off)
                width_win, height_win = int(win.width), int(win.height)

                # Step 1: Extract chunk and save as JPG
                chunk_array = src.read(window=win)
                chunk_img = np.moveaxis(chunk_array, 0, -1)  # Convert CHW to HWC

                # Fix: Drop alpha channel if present
                if chunk_img.shape[2] == 4:
                    chunk_img = chunk_img[:, :, :3]

                temp_chunk_jpg = os.path.join(output_directory, f"chunk_img_{left}_{top}.jpg")
                imageio.imwrite(temp_chunk_jpg, chunk_img)


                # Step 2: Generate mask on the JPG
                temp_mask_jpg = os.path.join(output_directory, f"chunk_mask_{left}_{top}.jpg")
                sam.generate(temp_chunk_jpg, temp_mask_jpg)

                # Step 3: Read and resize predicted mask
                mask = cv2.imread(temp_mask_jpg, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Failed to read mask: {temp_mask_jpg}")
                mask_resized = cv2.resize(mask, (width_win, height_win), interpolation=cv2.INTER_NEAREST)
                binary_mask = (mask_resized > 127).astype(np.uint8) * 255

                # Insert binary mask chunk into full-size mask
                final_mask[top:top + height_win, left:left + width_win] = binary_mask

                # Cleanup temp files
                os.remove(temp_chunk_jpg)
                os.remove(temp_mask_jpg)

            # Skeletonization (optional)
            skeleton = skeletonize(final_mask // 255).astype(np.uint8) * 255

            # Save final binary mask
            with rasterio.open(output_mask_path, 'w', **profile) as dst:
                dst.write(skeleton, 1)

            # Convert mask to vector
            sam.tiff_to_gpkg(output_mask_path, output_vector, simplify_tolerance=None)

            tqdm.write(f"✅ Saved final mask and vector for {filename}")


# === CONFIGURATION ===
input_directory = "/home/jazzy/sem-farm/data"
output_directory = "/home/jazzy/sem-farm/SEGMENTATION/SamGeo/output"
checkpoint_path = "/home/jazzy/sem-farm/SEGMENTATION/SamGeo/model/sam_vit_h_4b8939.pth"
os.makedirs(output_directory, exist_ok=True)

# === RUN ===
process_tif_directory_chunked(input_directory, output_directory, checkpoint_path, no_of_files=1)
