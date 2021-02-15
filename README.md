# Virtual Greenscreen

This project takes a webcam stream and applies a learning-based segmentation algorithm to create a greenscreen effect.   
The resulting image is provided as a virtual webcam.    
It utilizes the U^2 Net segmentation model.

## Requirements

- OBS: https://obsproject.com/
- obs-virtual-cam:
https://github.com/CatxFish/obs-virtual-cam/releases
- CUDA 10.2: https://developer.nvidia.com/cuda-10.2-download-archive
- Install python libraries:
`pip install -r requirements.txt` <br><br>
#### Download models:
1. Big accurate model: <br>
https://drive.google.com/file/d/1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P
2. Small fast model: <br>
https://drive.google.com/file/d/1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy
3. Put big model in pretrained/u2net_human_seg
4. Put small model in pretrained/u2netp


## Usage

1. `python virtual_greenscreen.py`
2. Select model
3. Run
4. In OBS, select "OBS-Camera" as camera
 
