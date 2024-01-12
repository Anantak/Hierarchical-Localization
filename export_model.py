import csv
from pycolmap import Reconstruction
import os

def export_reconstruction_to_csv(reconstruction_path, csv_file_path):
    # Load the reconstruction
    reconstruction = Reconstruction()
    reconstruction.read(reconstruction_path)

    # Create and open the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Qvec", "Tvec"])

        # Sort the images by name
        sorted_images = sorted(reconstruction.images.items(), key=lambda x: x[1].name)

        # Write the data to the CSV file
        for _, image in sorted_images:
            qvec = image.qvec
            tvec = image.tvec
            writer.writerow([image.name, qvec, tvec])

# Set the path to your COLMAP reconstruction directory and the desired CSV file path
reconstruction_path = "/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/testtrack/models"
csv_file_path = "/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/testtrack/models/output.csv"

# Run the function
export_reconstruction_to_csv(reconstruction_path, csv_file_path)

print("CSV file has been created at:", csv_file_path)
