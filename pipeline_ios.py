#!/usr/bin/env python
# coding: utf-8

# In[8]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

from pathlib import Path
from pprint import pformat

from hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval, pairs_from_ios_poses
from hloc import colmap_from_nvm, reconstruction, triangulation, localize_sfm, visualization

from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d


# # Pipeline for ANA image data

# ## Setup
# Here we declare the paths to the dataset, the reconstruction and localization outputs, and we choose the feature extractor and the matcher. 

# In[9]:


images = Path('datasets/office/frames/')
outputs = Path('outputs/office/')

sfm_pairs = outputs / 'pairs-db-covis20.txt'  # top 20 most covisible in SIFT model
loc_pairs = outputs / 'pairs-query-netvlad20.txt'  # top 20 retrieved by NetVLAD
reference_sfm = outputs / 'sfm_superpoint+superglue'  # the SfM model we will build
results = outputs / 'Aachen_hloc_superpoint+superglue_netvlad20.txt'  # the result file
sfm_dir = outputs / 'sfm'

# list the standard configurations available
# print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
# print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')


# In[3]:


# pick one of the configurations for image retrieval, local feature extraction, and matching
# you can also simply write your own here!
retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs['disk']
matcher_conf = match_features.confs['disk+lightglue']


# ## Extract local features for database and query images

# In[4]:


features = extract_features.main(feature_conf, images, outputs)


# The function returns the path of the file in which all the extracted features are stored.

# In[5]:


# This is the file where the features were saved
print(f'Features were exported to file: {features}')

# How can we plot the exported features?
from hloc.utils.io import list_h5_names, read_image, get_keypoints
from hloc.utils.viz import plot_images, plot_keypoints

sample_image_names_list = list_h5_names(features)

sample_idx = 3
sample_image_name = sample_image_names_list[sample_idx] #'db/1931.jpg'
sample_image_path = images / sample_image_name
sample_image = read_image(sample_image_path)
sample_image_kps = get_keypoints(features, sample_image_name)

plot_images([sample_image])
plot_keypoints([sample_image_kps])


# ## Generate pairs for the SfM reconstruction
# Instead of matching all database images exhaustively, we exploit the existing SIFT model to find which image pairs are the most covisible. We first convert the SIFT model from the NVM to the COLMAP format, and then do a covisiblity search, selecting the top 20 most covisibile neighbors for each image.

# In[6]:


# Define the file paths
poses_file_path = Path.home() / 'Anantak/Pipelines/Hierarchical-Localization/datasets/office/poses.txt'
rotation_threshold = 30

# Call the main function of your script
pairs_from_ios_poses.main(
    poses_file=poses_file_path,
    output=sfm_pairs,
    num_matched=8000,
    rotation_threshold=rotation_threshold
)


# ## Match the database images

sfm_matches = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)


print(f'sfm matches were saved in file {sfm_matches}')

# In[ ]:


sfm_dir = Path('/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/datasets/office/frames/')
images_path = Path('/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/datasets/office/frames/')
sfm_pairs_path = Path('/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/office/pairs-db-covis20.txt')
features_path = Path('/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/office/feats-disk.h5')
matches_path = Path('/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/office/feats-disk_matches-disk-lightglue_pairs-db-covis20.h5')

model = reconstruction.main(
    sfm_dir=sfm_dir,
    image_dir=images_path,
    pairs=sfm_pairs_path,
    features=features_path,
    matches=matches_path
    )

