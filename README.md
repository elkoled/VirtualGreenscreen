# Virtual Greenscreen

This project takes a webcam stream and applies a learning-based segmentation algorithm to create a greenscreen effect.   
The resulting image is provided as a virtual webcam.    
It utilizes the U^2 Net segmentation model.

## Required tools

- OBS
- obs-virtual-cam:
https://github.com/CatxFish/obs-virtual-cam/releases
- Download models:
1. Big accurate model: https://drive.google.com/file/d/1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P/view?usp=sharing
2. Small fast model: https://drive.google.com/file/d/1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy/view?usp=sharing
3. Put big model in pretrained/u2net_human_seg
4. Put small model in pretrained/u2netp

## Required libraries

- Python 3.8  
- numpy 1.20.1  
- PyTorch 1.7.1  
- torchvision 0.8.2  
- python-opencv 4.5.1.48
- Install with: pip install -r requirements.txt

## Usage

1. `python virtual_greenscreen.py`
2. Select model
3. Run
4. In OBS, select "OBS-Camera" as camera
