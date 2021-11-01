# Code from OpenCV Optical Flow Demo #
import os
import pickle
import cv2 as cv
import numpy as np
from pathlib import Path


def extractFeatures(cur, prev, delta):
    # Tangential Velocity
    tanVel = np.zeros((cur.shape[0], 3))
    tanVel[:,0] = cur[:,0]
    tanVel[:,1] = cur[:,1]
    tanVel[:,2] = delta
    tanVelMag = np.sqrt(np.square(cur[:,0]) + np.square(cur[:,1]) + delta ** 2)

    # Acceleration
    accel = np.zeros((cur.shape[0], 3))
    accel[:,0] = cur[:,0] - prev[:,0]
    accel[:,1] = cur[:,1] - prev[:,1]

    # Curvature
    curv = np.zeros((cur.shape[0]))
    curv = np.divide(np.linalg.norm(np.cross(tanVel, accel, 1, 1)), np.power(tanVelMag, 3))

    # Radius of Curvature
    rad = np.divide(np.ones((cur.shape[0])), curv)

    # Angular Velocity
    angVel = np.divide(tanVelMag, rad)

    return np.mean(tanVelMag), np.mean(curv), np.mean(rad), np.mean(angVel)


def processVideo(vid, savePath, filename):
    cap = cv.VideoCapture(cv.samples.findFile(vid))
    fps = cap.get(cv.CAP_PROP_FPS)
    delta = 1 / fps
    ret, frame1 = cap.read()
    prev = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    prevFlow = np.zeros((prev.shape[0], prev.shape[1], 2))
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    
    writing = False
    frameCount = 0
    while(cap.isOpened()):
        if (not writing and frameCount % 60 == 0):
            writing = True
            featureData = [1, []] if filename[0:3] == "bio" else [-1, []]
            size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
            normalWrite = cv.VideoWriter(savePath + filename + "_" + str(frameCount // 60) + "_vid.avi", cv.VideoWriter_fourcc(*'MJPG'), fps, size)
            debugWrite = cv.VideoWriter(savePath + filename + "_" + str(frameCount // 60) + "_debug.avi", cv.VideoWriter_fourcc(*'MJPG'), fps, size)
        elif (frameCount % 60 == 0):
            print("vid saved!")
            writing = False
            with open(savePath + filename + "_" + str((frameCount // 60) - 1) + '.pkl', 'wb') as file:
                pickle.dump(featureData, file)
            normalWrite.release()
            debugWrite.release()
            # Save that shit

        ret, frame2 = cap.read()
        if ret == False:
            break
        cur = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
        
        # Get dense optical flow
        curFlow = cv.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv.cartToPolar(curFlow[...,0], curFlow[...,1])
        mag = np.array(mag , dtype=np.uint8)

        # Apply thresholding 
        T, motionMask = cv.threshold(mag, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        motionMask = cv.GaussianBlur(motionMask, (25,25), cv.BORDER_DEFAULT) # Blur
        motionMask[motionMask > 0] = 255

        # Find motion contours
        contours, _ = cv.findContours(image=motionMask, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

        ## ??? Add better grouping here ??? ##
        ## Paper talks about perceptual grouping - I didn't implement it exactly like they did ##

        # Get largest contour and mask
        maxContour = max(contours, key=cv.contourArea)
        regionMask = np.zeros(motionMask.shape)
        cv.drawContours(image=regionMask, contours=[maxContour], contourIdx=0, color=255, thickness=-1)

        ## ??? Create convex hull ??? ##
        ## Decided not to do this, because the paper did not mention it ##
        ## This would also run the risk of including a very inconsistant number of pixels without any motion ##

        # Extract Features
        prevRegionFlow = prevFlow[np.where(regionMask == 255)[0],np.where(regionMask == 255)[1],:]
        curRegionFlow = curFlow[np.where(regionMask == 255)[0],np.where(regionMask == 255)[1],:]
        [tanVel, curv, rad, angVel] = extractFeatures(curRegionFlow, prevRegionFlow, delta)

        if writing:
            normalWrite.write(frame2)
            cv.drawContours(image=frame2, contours=[maxContour], contourIdx=0, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
            debugWrite.write(frame2)
            featureData[1].append([tanVel, curv, rad, angVel])

        ## For debugging ##
        # mag, ang = cv.cartToPolar(curFlow[...,0], curFlow[...,1])
        # hsv[...,0] = ang*180/np.pi/2
        # hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        # bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        # cv.drawContours(image=bgr, contours=[maxContour], contourIdx=0, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
        # cv.imshow('frame2',bgr)
        # k = cv.waitKey(30) & 0xff
        # if k == 27:
        #    break
        # elif k == ord('s'):
        #     cv.imwrite('opticalfb.png',frame2)
        #     cv.imwrite('opticalhsv.png',bgr)
        prev = cur
        prevFlow = curFlow
        frameCount += 1
    
    cap.release()
    cv.destroyAllWindows()

# For each video save clips of 60 frames - w/ 60 frames in between - Save file with nparray of extracted info for each frame
# Save outlined contour - delete bad examples
savePath = "../../training_data_done/"
findPath = "../../baseline_trimmed_data/"
for filename in os.listdir(findPath):
    print("file", filename)
    if filename.endswith(".mov"):
        if Path(filename).stem == "bio_game_1":
            continue
        processVideo(findPath + filename, savePath, Path(filename).stem)