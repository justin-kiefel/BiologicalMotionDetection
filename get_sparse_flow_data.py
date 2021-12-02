import numpy as np
import cv2 as cv
import os


cap = cv.VideoCapture(cv.samples.findFile("TrimmedData1/bio_cat_1_2.mov"))
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 200,
                qualityLevel = 0.1,
                minDistance = 7,
                blockSize = 3 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
            maxLevel = 2,
            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, frame1 = cap.read()
gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
ret, frame2 = cap.read()
gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
initial_flow = cv.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, _ = cv.cartToPolar(initial_flow[...,0], initial_flow[...,1])
mag = np.array(mag , dtype=np.uint8)
blur1 = cv.GaussianBlur(mag,(25,25),cv.BORDER_DEFAULT)
thr, thresh1 = cv.threshold(mag,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
blur2 = cv.GaussianBlur(thresh1,(31,31),cv.BORDER_DEFAULT)
th, motionMask = cv.threshold(blur2, 1, 255, cv.THRESH_BINARY);

# create hull array for convex hull points
hull = []




# Identify motion regions
contours, _ = cv.findContours(image=motionMask, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
contours = contours if (len(contours) > 0) else []
hulls = []
for i in range(len(contours)):
    hulls.append(cv.convexHull(contours[i], False))
regionMask = np.zeros(motionMask.shape)
cv.drawContours(regionMask, hulls, contourIdx=-1, color=255, thickness=-1)
#cv.drawContours(image=regionMask, contours=contours, contourIdx=-1, color=255, thickness=-1)
regionMask = regionMask.astype(np.uint8)

cv.imshow("ahhh", regionMask)
cv.waitKey(0)

p0 = cv.goodFeaturesToTrack(gray2, mask = regionMask, **feature_params)
# Create a mask image for drawing purposes
old_gray = gray2
old_frame = frame2
mask = np.zeros_like(old_frame)
while(1):
    ret,frame = cap.read()
    print(frame, type(frame), frame.shape)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    #print(type(old_gray), old_gray.shape, old_gray.dtype)
    #print(type(frame_gray), frame_gray.shape, frame_gray.dtype)
    #print(type(p0), p0.shape)
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)
    cv.imshow('frame',img)
    cv.waitKey(0)
    #k = cv.waitKey(30) & 0xff
    #if k == 27:
    #    break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
