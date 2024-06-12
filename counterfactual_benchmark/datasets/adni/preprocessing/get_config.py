from pathlib import Path
import os

def get_config_dict():
    config = {}
    config["data_path"] = Path(r'./preselected_data/')
    config["re_process"] = False

    resolution_mm = 1
    config["reference_atlas_location"] = Path(f'{os.environ["FSLDIR"]}/data/standard/MNI152_T1_{resolution_mm}mm_brain.nii.gz')
    config["axial_size"] = 180
    config["central_crop_along_z"] = True
    config["central_crop_size"] = 30 # size of final image along z axis (number of slices to save)
    config["save_2d"] = True
    config["remove_nii"] = True
    config["subject_limit"] = -1 # limit number of subjects to preprocess (-1 to disable)
    return config
