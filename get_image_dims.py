from PIL import Image
import os

def get_image_dimensions(image_path):
    """ Returns the dimensions of the image at the given path. """
    with Image.open(image_path) as img:
        return img.size  # (width, height)

# Example usage
image_path = '/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/datasets/testtrack/path_queries/out-1.jpg'  # Replace with your image path
width, height = get_image_dimensions(image_path)
print(f"Width: {width}, Height: {height}")
