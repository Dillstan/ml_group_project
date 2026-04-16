import glob
import os

# Define your root folder
# Use os.path.abspath to avoid ".." confusion later
root_dir = os.path.abspath("../../../../ml_project_files/imdb_crop")

# The "**" tells it to look in all subfolders
# recursive=True is required for "**" to work
all_image_paths = glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)

print(f"Found {len(all_image_paths)} images.")
#print(all_image_paths[:5])

#with open('./logs/db_reg_imgpaths', 'w') as f:
#    for image_path in all_image_paths:
#        f.write(image_path+'\n')

