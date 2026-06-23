from PIL import Image

def convert_tiff_to_jpg(input_tiff, output_jpg, quality=100):
    with Image.open(input_tiff) as img:
        img = img.convert("RGB")  # Convert to RGB mode (TIFF might have transparency)
        img.save(output_jpg, "JPEG", quality=quality)

# Example usage
convert_tiff_to_jpg("/home/jazzy/sem-farm/data/test55.tif", "/home/jazzy/sem-farm/data/test55.jpg")
