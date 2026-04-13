
<!-- Banner / Logo (optional) -->
<p align="center">
  <img src="assets/banner.png" alt="Project Banner" width="140">
</p>

<h1 align="center">Rin</h1>

<p align="center">
  Good luck
</p>

<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3">
  </a>
</p>


---

## Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [Parameters](#parameters)
- [Demo](#demo)

## Features
- Feature detection
- RANSAC allignment (WIP)
- Fast Stitching preview + export (PNG/TIFF)
- GUI for plotting SPM Data

## Quick Start

> **Prereqs:** Python + libraries
### Install requirements 
```bash
pip install -r requirements.txt
```
## Script divided into two parts, recognition and data plotting

<div style="display: flex; justify-content: space-between;">

<div style="flex: 1; margin-right: 10px;">
  
### Recognition
- Run <span class="py_files">cv2_ransac.py</span> or <span class="py_files">scikit_ransac.py</span> (PNG or TIFF)
- Into main write parameters, such as folder path(<span class="input">TILE_DIR</span>)  and filtering options([Parameters](#parameters)). For now you can use default
- Output  <span class="files"> match_offsets.txt </span>
 - Fast reconstruction could be done with <span class="py_files">fast_recover.py</span> (<span class="input">TILE_DIR</span>)
 - You can run <span class="py_files">agparse.py</span> terminal and parse arguments ([exapmle](#example_arg))
</div>
<div style="flex: 1; margin-left: 10px;">

### Plotting
- Run <span class="py_files">gwy_plot.py</span> 
- Choose folder which consist of your gwy files and <span class="files"> match_offsets.txt </span>
- If it wasn't alligned for the first time, open folder from GUI (<span class="button">Choose</span>) and <span class="button">Reload</span>
- <span class="button">Apply tile unit</span> and <span class="button">tile size</span> and after proceed with everything else
- For x/y tile movement choose tile (either on screen/dropdown) move with <span class = "keyboard">arrows</span>(1px) or <span class = "keyboard">Shift+arrows</span>(50px)
- For "z" movement choose <span class="button">value step and step units </span> in GUI and press <span class = "keyboard">W/S</span> up/down
</div>

</div>


## example_arg

```bash
python3 agparse.py --methods sift --tile_dir "/Volumes/T7/" --matching_ratio 0.4 --matching_angle 10 --min_matches 4 --len_gap 20 --selected_method sift --ransac_enabled --three_point_level --ransac_thresh 5.0 --ransac_min_inliers 4 --preprocess_enabled true
```
## Parameters

```bash 
--methods sift, orb, dense_sift # what method to use, can be several at same time
--tile_dir # folder where script looks to find matches
--matching_ratio # what percent on green lines needed to accept match
--matching_angle # max angle deviation
--min_matches # min amount of keypoints
--len_gap #length from most common cluster deviation
--selected_method #method which will write match_offsets.txt 
--ransac_enabled #enable RANSAC
--ransac_thesh # RANSAC reprojection threshold
--ransac_min_inliers # minimum inliers for RANSAC
--preprocess_enabled # preprocess image before searching for keypoins 
```
Same goes into fuction if you want to use it
</pre>
## Demo

<div style="display: flex; justify-content: space-between;">

<div style="flex: 1; margin-right: 10px;">

### Accepted Matches
<p align="center">
    <img src="assets/acc1.png" alt="Accepted Matches" width="300">
</p>
<p align="center">
    <img src="assets/acc2.png" alt="Accepted Matches" width="300">
</p>

</div>

<div style="flex: 1; margin-left: 10px;">

### Rejected Matches
<p align="center">
    <img src="assets/rej1.png" alt="Rejected Matches" width="300">
</p>
<p align="center">
    <img src="assets/rej2.png" alt="Rejected Matches" width="300">
</p>
</div>

</div>

As you can see you can debug it using colors of lines: <font color = "green">Green</font> - line accepted, <font color = "red">Red</font> - angle is out of bounds, <font color = "purple">Purple</font> - length is out of bounds, <font color = "blue">Blue</font> - shape check

In case script failde to find some matches, or too strict filters were applied you can check <span class="files">stitching_log.log</span>, to find missing values. There is always at least one that will be correct. And later write length and angle into  <span class="files">match_offsets.txt</span> manually.


Usefull terminal command to convert all files into .gwy file
```bash
   mkdir -p gwy                                                                 
for f in *.tiff; do 
  [ -e "$f" ] || continue
  gwyddion --convert-to-gwy="gwy/${f%.*}.gwy" "$f"
done
```





