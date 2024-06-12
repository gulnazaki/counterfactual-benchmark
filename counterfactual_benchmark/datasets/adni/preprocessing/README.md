# ADNI preprocessing

The codebase in this folder is based on this [fork](https://github.com/nacer-benyoub/adni_preprocessing/tree/refactor/update-fslinstaller) of this [github repo](https://github.com/SANCHES-Pedro/adni_preprocessing) which is in turn based on [Tian Xia's code](https://github.com/xiat0616).
Huge thanks to everyone involved!

## Dependencies

```
python3 -m pip install simpleitk pandas
python2 fsl_ubuntu/fslinstaller.py
```

## Preprocessing
```
python3 image_preselection.py --csv ADNI1_Complete_1Yr_1.5T_<INSERT_DOWNLOAD_DATE>_.csv
python3 move_selected_images.py
python3 main.py
```

## The processing steps done are the following

1. Reorientation (FSL fslreorient2std)
2. Cropping (FSL robustfov)
3. Brain Extraction (FSL bet)
4. Atlas Affine Registration (FSL flirt)
    1. Resolution (1 or 2 mm) can be chosen at this step
5. Structure segmentation and bias correction (FSL fast)
6. Intensity normalization between [-1,1]
    1. Clip intensity at 99.5% of max values (numpy.percentile)
7. Central slice cropping. Default to 60mm around the centre along the z axis.