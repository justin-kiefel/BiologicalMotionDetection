import numpy as np
import os
import cv2 as cv
import tensorflow as tf
import keras

from numpy.core.numeric import count_nonzero
from sklearn.kernel_ridge import KernelRidge 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.model_selection import LeaveOneOut


def extractKinematics(cur, prev, delta):
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


def extractFeatures(data, delta, one_region=True):
    """
    Input: data - Optical flow data (T x W x H x 2) dtype = np.int8
           delta - Time difference between frames
           one_region - True if kinematics averaged for max motion region only
    Output: kinematics - Extracted average motion kinematics (T x 4)
    """
    prev = np.zeros((data.shape[1], data.shape[2], 2))
    kinematics = np.zeros((data.shape[0], 4))
    
    for t in range(data.shape[0]):
        cur = data[t,:,:,:].astype(np.float32)

        # Adaptive Threshold for Motion - ???try fixed threshold later???
        mag, _ = cv.cartToPolar(cur[...,0], cur[...,1])
        mag = np.array(mag , dtype=np.uint8)
        blur = cv.GaussianBlur(mag,(13,13),cv.BORDER_DEFAULT)
        thr, motionMask = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        # Identify motion regions
        contours, _ = cv.findContours(image=motionMask, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
        regionMask = np.zeros(motionMask.shape)
        if (one_region):
            maxContour = [max(contours, key=cv.contourArea)] if (len(contours) > 0) else []
            cv.drawContours(image=regionMask, contours=maxContour, contourIdx=0, color=255, thickness=-1)
        else:
            contours = contours if (len(contours) > 0) else []
            cv.drawContours(image=regionMask, contours=contours, contourIdx=-1, color=255, thickness=-1)

        # Get average kinematics for all non zero regions or main region
        prevRegionFlow = prev[np.where(regionMask == 255)[0],np.where(regionMask == 255)[1],:]
        curRegionFlow = cur[np.where(regionMask == 255)[0],np.where(regionMask == 255)[1],:]
        kinematics[t,:] = np.array(extractKinematics(curRegionFlow, prevRegionFlow, delta))

        prev = cur

    return kinematics


def trainKRR(deltas, raw_x, raw_y):
    X = np.zeros((len(raw_x), 4 * len(deltas)))
    for i, x in enumerate(raw_x):
        for j, delta in enumerate(deltas):
            X[i,j*4:(j+1)*4] = x[59-delta,:]
    X = scale(X)
    Y = np.array(raw_y)

    cv = LeaveOneOut()
    # enumerate splits
    y_true, y_pred = list(), list()

    for train_ix, test_ix in cv.split(X):
        # split data
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = Y[train_ix], Y[test_ix]
        # fit model
        krr = KernelRidge(alpha=1.0, kernel='rbf')

        param_grid = [
        {'alpha': [0.1, 1.0, 10.0, 100.0, 10000.0, 1000000.0], 'gamma': [0.001, 0.01, 0.1, 1.0], 'kernel': ['rbf']},
        ]

        search_krr = GridSearchCV(krr, param_grid, cv=5)
        best_krr = search_krr.fit(X_train, y_train)

        yhat = best_krr.predict(X_test)
        y_true.append(y_test[0])
        y_pred.append(yhat[0])
    # calculate accuracy

    y_pred = np.array(y_pred)
    y_pred[y_pred < 0] = -1
    y_pred[y_pred >= 0] = 1
    acc = accuracy_score(y_true, y_pred)
    print('Accuracy: %.3f' % acc)


def baseline(raw_x, raw_y):
    delta_list = [[2], [10], [20], [30], [40], [50], [60], [2,30,60]]
    resDict = {}
    for i,deltas in enumerate(delta_list):
        resDict[i] = trainKRR(deltas, raw_x, raw_y)


def baseLSTM(raw_x, raw_y):
    return 


def main():
    # Modes = ['baseline', 'baseline+lstm']
    mode = 'baseline+lstm'
    data_path = 'ProcessedData/'
    fps = 30
    one_region = False
    delta = 1/fps

    # Load Data
    raw_x = []
    raw_y = []
    for filename in os.listdir(data_path):
        if (filename[-4:] == ".npz"):
            print("filename: ", filename)
            data = np.load(open(data_path + filename, 'rb'))['arr_0']
            raw_x.append(extractFeatures(data, delta, one_region=one_region))
            
            if "bio" == filename[0:3]:
                raw_y.append(1)
            else:
                raw_y.append(-1)
    
    if mode == 'baseline':
        baseline(raw_x, raw_y)
    elif mode == 'baseline+lstm':
        baseLSTM(raw_x, raw_y)
    else:
        raise NotImplementedError


main()