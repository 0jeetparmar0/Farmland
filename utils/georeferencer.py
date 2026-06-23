import rasterio
from rasterio.transform import from_bounds
import numpy as np
from PIL import Image
import re

# File paths
input_jpg = "/home/jazzy/sem-farm/Spatial_temporal/data/025.jpg"  # Your JPG image
output_tiff =  "/home/jazzy/sem-farm/utils/025.tif" # Output GeoTIFF

# Step 1: Define 4 coordinates in DMS format
coordinates_dms = [
    ("23°10'34.24\"N", "72°37'30.37\"E"),  # Bottom-left
    ("23°10'34.29\"N", "72°37'46.67\"E"),   # Bottom-right
    ("23°10'44.67\"N", "72°37'46.67\"E"),   # Top-right
    ("23°10'44.63\"N", "72°37'30.38\"E")   # Top-left
]

# Function to convert DMS to Decimal Degrees
def dms_to_decimal(dms):
    match = re.match(r"(\d+)°(\d+)'([\d.]+)\"([NSEW])", dms)
    if not match:
        raise ValueError(f"Invalid DMS format: {dms}")
    
    degrees, minutes, seconds, direction = match.groups()
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    
    # Handle direction (N/E positive, S/W negative)
    if direction in ["S", "W"]:   
        decimal = -decimal
    
    return decimal

# Convert all coordinates to decimal degrees
coordinates = [(dms_to_decimal(lat), dms_to_decimal(lon)) for lat, lon in coordinates_dms]

# Extract bounding box (min_x, min_y, max_x, max_y)
min_x = min(coord[1] for coord in coordinates)  # Minimum longitude
max_x = max(coord[1] for coord in coordinates)  # Maximum longitude
min_y = min(coord[0] for coord in coordinates)  # Minimum latitude
max_y = max(coord[0] for coord in coordinates)  # Maximum latitude

# Step 2: Open JPG image using PIL
image = Image.open(input_jpg)
image_array = np.array(image)

# Step 3: Compute transformation from bounding box
transform = from_bounds(min_x, min_y, max_x, max_y, image_array.shape[1], image_array.shape[0])

# Step 4: Save as GeoTIFF with georeferencing
with rasterio.open(
    output_tiff,
    'w',
    driver='GTiff',
    height=image_array.shape[0],
    width=image_array.shape[1],
    count=3,  # RGB Channels
    dtype=image_array.dtype,
    crs="EPSG:4326",  # WGS 84 Coordinate Reference System
    transform=transform
) as dst:
    for i in range(3):  # Save R, G, B bands separately
        dst.write(image_array[:, :, i], i + 1)

print("✅ GeoTIFF successfully created:", output_tiff)