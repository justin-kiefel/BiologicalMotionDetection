# Code from OpenCV Optical Flow Demo #
import os
import cv2 as cv
import numpy as np
from pathlib import Path


def processVideo(vid, savePath, filename):
    cap = cv.VideoCapture(cv.samples.findFile(vid))
    ret, frame1 = cap.read()
    scale_percent = 33.3333333333 # percent of original size
    width = int(frame1.shape[1] * scale_percent / 100)
    height = int(frame1.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame1 = cv.resize(frame1, dim)
    prev = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    flow_data = np.zeros((60,frame1.shape[0], frame1.shape[1], 2)) 
    
    writing = False
    frameCount = 0
    while(cap.isOpened() and frameCount < 60):
        ret, frame2 = cap.read()


        if ret == False:
            break
        frame2 = cv.resize(frame2, dim)
        cur = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)

        # Get dense optical flow
        flow = cv.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = flow.astype(np.int8)
        
        # For Debugging #
        #flow = flow.astype(np.float32)
        #mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        #hsv = np.zeros_like(frame1)
        #hsv[...,1] = 255
        #hsv[...,0] = ang*180/np.pi/2
        #hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        #bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        #cv.imshow('frame2',bgr)
        #cv.waitKey(0)

        # Convert datatype
        flow_data[frameCount, :, :, :] = flow
        frameCount += 1
    
    with open(savePath + filename + '_flow.npz', 'wb') as file:
        np.savez_compressed(file, flow_data)

    cap.release()
    cv.destroyAllWindows()

# For each video save clips of 60 frames - w/ 60 frames in between - Save file with nparray of extracted info for each frame
# Save outlined contour - delete bad examples
savePath = "ProcessedDataNew/"
findPath = "TrimmedData1/"
for filename in os.listdir(findPath):
    print("file", filename)
    if filename.endswith(".mov"):
        processVideo(findPath + filename, savePath, Path(filename).stem)

savePath = "ProcessedDataNew/"
findPath = "TrimmedData2/"
for filename in os.listdir(findPath):
    print("file", filename)
    if filename.endswith(".mov"):
        processVideo(findPath + filename, savePath, Path(filename).stem)