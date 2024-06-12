import pandas as pd
from pathlib import Path
import os

"""
Move the selected images to another directory called `preselected_data`
>>> The images have to be in a directory called `raw_data` which has the following structure:
    raw_data/<subject_id>/<preprocessing>/<date>/<acquisition_id>/<file_name>.nii
"""

raw_data_dir = r'./raw_data'

# Read the images IDs
print('Loading the `final_image_ids.csv` file...')
df = pd.read_csv('final_image_ids.csv')
assert len(df.columns) == 1, f'`final_image_ids.csv` should have only one column. Found {len(df.columns)}'
print('Done')

# Move the final images to 'preselected' directory
for folder, dirs, filenames in os.walk(raw_data_dir):
    dirname = Path(folder).name

    if dirname.startswith('I') and (dirname in df.iloc[:, 0].values):
        # folder is an image ID and that ID is in the final selected IDs

        for filename in filenames: # should exist only one but just in case
            src_image_path = Path(os.path.join(folder, filename))
            dst_image_path = Path(os.path.join(folder, filename).replace("raw", "preselected"))
            print(f'Moving the image {dirname} from {src_image_path} to {dst_image_path}...')
            dst_image_path.parent.mkdir(parents=True, exist_ok=True)
            src_image_path.replace(dst_image_path)
            print('Done')

print('\nAll the selected images were moved')