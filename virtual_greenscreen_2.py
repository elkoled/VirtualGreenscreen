import os
from pickle import TRUE
import cv2
import numpy as np
import time 
from PIL import Image
import pyvirtualcam
from torchvision.transforms import ToTensor
import torch
from torch.autograd import Variable
import torch.nn.functional as torchNN
from threading import Thread, Lock

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small size version 4.7MB

# set webcam resolution and target fps
# high resolutions requires much VRAM
width = 1280
height = 720
target_fps = 30

# inference scale, 1.0 = full image resolution used for inference
inference_scale = 0.25


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def checkCuda():
    if torch.cuda.is_available():
        print("CUDA GPU found")
    else:
        print("No CUDA GPU found, using CPU...")
    


# ----------- Utility classes -------------


# A wrapper that reads data from cv2.VideoCapture in its own thread to optimize.
# Use .read() in a tight loop to get the newest frame
class Camera:
    def __init__(self, device_id=0, width=1280, height=720):
        self.capture = cv2.VideoCapture(device_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.success_reading, self.frame = self.capture.read()
        self.read_lock = Lock()
        self.thread = Thread(target=self.__update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __update(self):
        while self.success_reading:
            grabbed, frame = self.capture.read()
            with self.read_lock:
                self.success_reading = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
        return frame
    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()

# An FPS tracker that computes exponentialy moving average FPS
class FPSTracker:
    def __init__(self, ratio=0.5):
        self._last_tick = None
        self._avg_fps = None
        self.ratio = ratio
    def tick(self):
        if self._last_tick is None:
            self._last_tick = time.time()
            return None
        t_new = time.time()
        fps_sample = 1.0 / (t_new - self._last_tick)
        self._avg_fps = self.ratio * fps_sample + (1 - self.ratio) * self._avg_fps if self._avg_fps is not None else fps_sample
        self._last_tick = t_new
        return self.get()
    def get(self):
        return self._avg_fps

# Wrapper for playing a stream with cv2.imshow(). It can accept an image and return keypress info for basic interactivity.
# It also tracks FPS and optionally overlays info onto the stream.
class Displayer:
    def __init__(self, title, width=None, height=None, show_info=True):
        self.title, self.width, self.height = title, width, height
        self.show_info = show_info
        self.fps_tracker = FPSTracker()
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        if width is not None and height is not None:
            cv2.resizeWindow(self.title, width, height)
    # Update the currently showing frame and return key press char code
    def step(self, image):
        fps_estimate = self.fps_tracker.tick()
        if self.show_info and fps_estimate is not None:
            message = f"{int(fps_estimate)} fps | {self.width}x{self.height}"
            cv2.putText(image, message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
        cv2.imshow(self.title, image)
        return cv2.waitKey(1) & 0xFF

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


# --------------- Main ---------------
checkCuda()

model_name='u2net_human_seg'  
model_dir = os.path.join(os.getcwd(), 'pretrained', model_name, model_name + '.pth')

model = U2NET(3,1).cuda().eval()
model.load_state_dict(torch.load(model_dir), strict=False)


cam = Camera(width=width, height=height)
dsp = Displayer('VirtualGreenscreen', cam.width, cam.height, show_info=TRUE)

def cv2_frame_to_cuda(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ToTensor()(Image.fromarray(frame)).unsqueeze_(0).cuda()

with torch.no_grad():
    while True:
        bgr = np.full((width, height, 3), (0,255,0), dtype=np.uint8)
        while True: # matting
            webcam_frame = cam.read()
            frame = cv2.resize(webcam_frame, None, fx=inference_scale, fy=inference_scale, interpolation=cv2.INTER_AREA)
            # start of Inference
            frame = np.array(frame).astype(np.float32) / 255.  # Scale
            frame = frame[:, :, [2, 1, 0]]  # Swap channels (to BGR)

            # create pytorch Variable
            frame = frame.transpose(2, 0, 1)  # Transpose
            inputs_test = frame[np.newaxis, ...]
            inputs_test = torch.FloatTensor(inputs_test)
            inputs_test = Variable(inputs_test.cuda())
            d1,d2,d3,d4,d5,d6,d7= model(inputs_test)
            pred = d1[:,0,:,:]
            pred = normPRED(pred)
            pred = pred.unsqueeze(0)

            mask = torchNN.interpolate(pred, size=(height,width), mode='bilinear', align_corners=False)
            # uncomment
            src = cv2_frame_to_cuda(webcam_frame)
            res = mask * src + (1 - mask) * torch.ones_like(src)
            res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
            # uncomment

            # test:
            # mask = mask.cpu()
            # src = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
            # #print(src.is_cuda)
            # src = ToTensor()(Image.fromarray(src)).unsqueeze_(0)
            # print(src.is_cuda)
            # res = mask * src + (1 - mask) * torch.ones_like(src)
            # print(res.is_cuda)
            # res = res.mul(255).byte().permute(0, 2, 3, 1).numpy()[0]
            #print(res.is_cuda)
            
            
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            key = dsp.step(res)
            if key == 27:
                exit()