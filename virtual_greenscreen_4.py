import os
import cv2
import numpy as np
import time 
from PIL import Image
import pyvirtualcam
import torch
import torch.nn.functional as torchNN
from torch.autograd import Variable

from model import U2NET,U2NETP # full size version 173.6 MB
#from model import U2NETP # small size version 4.7MB

# set webcam resolution and target fps
# high resolutions requires much VRAM
width = 1280
height = 720
target_fps = 30

# inference scale, 1.0 = full image resolution used for inference
inference_scale = 0.3


def main():
    checkCuda()

    #TODO Make model selectable in cmd
    ###### 1. select model ######
    model_name='u2net_human_seg'
    #model_name='u2netp'
    
    model_dir = os.path.join(os.getcwd(), 'pretrained', model_name, model_name + '.pth')

    ###### 2. setup webcam ######
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    image_shape = (width, height)
    cap.set(3, width)
    cap.set(4, height)

    ###### 3. load pretrained model ######
    if(model_name=='u2net_human_seg'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    ###### 4. inference loop ######
    # fps counter variables
    new_frame_time = 0
    prev_frame_time = 0

    background = np.full((width,height,3), (0,255,0), dtype=np.uint8)
    background = background.swapaxes(0,1)

    with pyvirtualcam.Camera(width=width, height=height, fps=target_fps) as cam:
        while True:
            # frame = np.zeros((cam.height, cam.width, 4), np.uint8) # RGBA
            # frame[:,:,:3] = cam.frames_sent % 255 # grayscale animation
            # frame[:,:,3] = 255
            new_frame_time = time.time()
            ret, webcam_frame = cap.read()
            frame = cv2.resize(webcam_frame, None, fx=inference_scale, fy=inference_scale, interpolation=cv2.INTER_AREA)

            # start of Inference
            frame = np.array(frame).astype(np.float32) / 255.  # Scale
            frame = frame[:, :, [2, 1, 0]]  # Swap channels (to BGR)

            # create pytorch Variable
            frame = frame.transpose(2, 0, 1)  # Transpose
            inputs_test = frame[np.newaxis, ...]
            inputs_test = torch.FloatTensor(inputs_test)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            d1,d2,d3,d4,d5,d6,d7= net(inputs_test)
            if __debug__:
                print("predict: %.2fms"%((time.time()-new_frame_time)*1000))
                time3 = time.time()
            # normalization
            pred = d1[:,0,:,:]
            pred = normPRED(pred)
            pred = pred.unsqueeze(0)
            res = torchNN.interpolate(pred, size=(height,width), mode='bilinear', align_corners=False)
            if __debug__:
                print("interpolate: %.4fms" %((time.time()-time3)*1000))
            time1 = time.time()
            res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
            if __debug__:
                print("gpu to cpu copy: %.4fms" %((time.time()-time1)*1000))
                time2 = time.time()
            res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)

            if __debug__:
                print("gray to rgb: %.4fms" %((time.time()-time2)*1000))
                time5 = time.time()
    
            webcam_frame = webcam_frame.astype(float)
            res = res.astype(float)/255
            #res = cv2.multiply(res, webcam_frame)/255
            #print(webcam_frame.shape)
            #print(res.shape)
            #print(background.shape)
            #res.squeeze()
            output = webcam_frame*res+(1.0-res)*background#cv2.addWeighted(webcam_frame,res,background,1.0-res,0)
            output = output/255

            if __debug__:
                print("alpha mask: %.2fms"%((time.time()-time5)*1000))
                time4 = time.time()

            cv2.imshow('Input', output)

            if __debug__:
                print("show image: %.2fms"%((time.time()-time4)*1000))
            

            # Calc and print FPS every 0.5s
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            print("FPS: %.2f" % fps)            


            #frame_web = np.zeros((cam.height, cam.width, 4), np.uint8) # RGBA
            #frame_web[:,:,:3] = cam.frames_sent % 255 # grayscale animation
            #frame_web[:,:,3] = 255

            #TODO Write to virtual camera
            #info = np.info(outImage.dtype)
            # outImage = outImage.astype(np.float64)
            # outImage = 255 * outImage
            # img = outImage.astype(np.uint8)

            # cam.send(img)
            # cam.sleep_until_next_frame()

            c = cv2.waitKey(1)
            if c == 27:
                break


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
    


if __name__ == "__main__":
    main()
