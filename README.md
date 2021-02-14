# Virtual Greenscreen

This project takes a webcam stream and applies a learning-based segmentation algorithm to create a greenscreen effect.   
The resulting image is provided as a virtual webcam.    
It utilizes the U^2 Net segmentation model.

## Required tools

- OBS
- obs-virtual-cam:
https://github.com/CatxFish/obs-virtual-cam/releases

## Required libraries

- Python 3.8  
- numpy 1.20.1  
- PyTorch 1.7.1  
- torchvision 0.8.2  
- python-opencv 4.5.1.48

## Usage

`python virtual_greenscreen.py`
1. Select model
2. Run
3. In OBS, select "OBS-Camera" as camera
