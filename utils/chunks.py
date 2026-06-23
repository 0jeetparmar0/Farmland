from PIL import Image
import os

def split_tiff_to_jpg_chunks(tiff_path, output_folder, chunk_size=256):
    # Open the TIFF image
    image = Image.open(tiff_path)
    width, height = image.size
    
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    chunk_count = 0
    
    # Loop through the image and extract 1024x1024 chunks
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            # Define the box for cropping
            box = (x, y, min(x + chunk_size, width), min(y + chunk_size, height))
            chunk = image.crop(box)
            
            # Save the chunk as a JPG file
            chunk_filename = os.path.join(output_folder, f"chunk_{chunk_count}.jpg")
            chunk.convert("RGB").save(chunk_filename, "JPEG")
            
            chunk_count += 1
    
    print(f"Saved {chunk_count} chunks in {output_folder}")

# Example usage
tiff_path = "/home/jazzy/sem-farm/Spatial_temporal/data/2025.tif"  # Replace with your TIFF file path
output_folder = "/home/jazzy/sem-farm/Spatial_temporal/spat_data1"  # Replace with your desired output directory
split_tiff_to_jpg_chunks(tiff_path, output_folder)
