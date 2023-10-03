import os
import random
import shutil
import argparse
import yaml

# Define source and destination directories
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
images_dir = os.path.join(project_dir, 'dataset/data/images')
dest_dir = os.path.join(project_dir, 'dataset/data/valimages')

# Parse command-line arguments to get the config file path
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='custom/og_config.yaml')
args = parser.parse_args()
config_file = os.path.join(project_dir, args.config)

# Load seed value from the config file
with open(config_file, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

seed = params.get('seed', 42)  # Use 42 as the default seed if not specified

# List all files in the source directory
all_items = os.listdir(images_dir)

# Randomly shuffle the list
random.shuffle(all_items)

# Select the first 1000 items
selected_items = all_items[:1000]

# Move the selected items to the destination directory
for item in selected_items:
    source_path = os.path.join(images_dir, item)
    dest_path = os.path.join(dest_dir, item)
    shutil.move(source_path, dest_path)

print("Moved 1000 random items to", dest_dir)
