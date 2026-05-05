# TEMscope
Automated segmentation and feature analysis of TEM images using contrast-based clustering.

This workflow enhances grayscale TEM images, segments them into contrast classes (K = 4), isolates the darkest domains, and quantifies feature size and morphology using metrics such as Feret diameter and equivalent diameter.

Designed for rapid, reproducible analysis and visualization with minimal user input.

## Key Features
- Automated segmentation (K-means, K = 4)
- Dark-feature extraction with dataset-level validation
- Morphological analysis (Feret, equivalent diameter, shape metrics)
- Publication-ready overlays and summary plots
- Batch processing of multiple images

## Usage
Run the notebook and place images in the `TEM_images/` folder.

## Output
- Segmented overlays with color maps
- Feature-level statistics
- Summary plots across samples


[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nrsarkerdevs/TEMscope/main?labpath=tem_seg_main.ipynb&force-rebuild=true)
