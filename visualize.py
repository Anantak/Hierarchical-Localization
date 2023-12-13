import os
import sys
import pycolmap
from hloc.utils.viz_3d import init_figure, plot_reconstruction

def main(model_path):
    # Load the reconstruction
    rec = pycolmap.Reconstruction()
    rec.read(model_path)  # No need to specify the extension

    # Initialize figure and plot reconstruction
    fig = init_figure()
    plot_reconstruction(fig, rec, points_rgb=True)

    # Show the figure
    fig.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_sfm.py <path_to_model_directory>")
        sys.exit(1)

    model_path = sys.argv[1]
    main(model_path)
