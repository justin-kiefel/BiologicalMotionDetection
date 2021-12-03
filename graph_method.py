from cv2 import MAT_CONTINUOUS_FLAG
from pathlib import Path
import numpy as np
import cv2 as cv
import os

from numpy.core.numeric import True_


# Search for areas with motion in first 5 frames
def getMotionMask(vidPath, start_frame=0):
    cap = cv.VideoCapture(vidPath)
    
    frame_cnt = 0
    while (frame_cnt < start_frame):
        ret, frame = cap.read()
        frame_cnt += 1

    ret, frame1 = cap.read()
    scale_percent = 33.33333333
    width = int(frame1.shape[1] * scale_percent / 100)
    height = int(frame1.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame1 = cv.resize(frame1, dim)
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    prev = gray1
    motionMask = np.zeros((height, width), dtype=np.uint8) # Adjust Dtype?
    for i in range(5):
        ret, frame = cap.read()
        frame = cv.resize(frame, dim)
        cur = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

        flow = cv.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv.cartToPolar(flow[...,0], flow[...,1])
        mag = np.array(mag , dtype=np.uint8)

        blur = cv.GaussianBlur(mag,(13,13),cv.BORDER_DEFAULT)
        thr, bin = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        blur2 = cv.GaussianBlur(mag,(23,23),cv.BORDER_DEFAULT)
        thr, bin2 = cv.threshold(blur2,0,255,cv.THRESH_BINARY)
        motionMask = cv.bitwise_or(bin2, motionMask)

    cap.release()
    cv.destroyAllWindows()
    return motionMask


def getGraphData(vidPath, k_max, start_frame):
    print(vidPath)

    # Calculate Motion Region to Search for Keypoints
    motionMask = getMotionMask(vidPath, start_frame=start_frame)

    # Search for Keypoints
    cap = cv.VideoCapture(vidPath)

    frame_cnt = 0
    while (frame_cnt < start_frame):
        ret, frame = cap.read()
        frame_cnt += 1
    ret, frame1 = cap.read()

    scale_percent = 33.33333333
    width = int(frame1.shape[1] * scale_percent / 100)
    height = int(frame1.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame1 = cv.resize(frame1, dim)
    prev = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    feature_params = dict(maxCorners = 500,qualityLevel = 0.15,minDistance = 4,blockSize = 7)
    lk_params = dict(winSize  = (15, 15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    p0 = cv.goodFeaturesToTrack(prev, mask = motionMask, **feature_params)
    tracks = [[(p[0,0], p[0,1])] for p in p0]
    tracks_dist = [0 for p in p0]
    finished_tracks = []
    finished_tracks_dist = []
    frame_cnt = 0

    while(ret and frame_cnt < 30):
        ret,frame = cap.read()
        if (not ret):
            break
        frame = cv.resize(frame, dim)
        cur = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        vis = frame.copy()

        if (len(tracks) > 0):
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            p1, _st, _err = cv.calcOpticalFlowPyrLK(prev, cur, p0, None, **lk_params)
            p0r, _st, _err = cv.calcOpticalFlowPyrLK(cur, prev, p1, None, **lk_params)
            cur_dists = np.sqrt(np.sum(np.square(p0-p1)[:,0,:], axis=1))
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            new_tracks_dist = []
            for tr, (x, y), good_flag, prev_dist, cur_dist in zip(tracks, p1.reshape(-1, 2), good, tracks_dist, cur_dists):
                if not good_flag:
                    finished_tracks.append(tr)
                    finished_tracks_dist.append(prev_dist)
                tr.append((x, y))
                new_tracks.append(tr)
                new_tracks_dist.append(cur_dist + prev_dist)
                cv.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
            tracks = new_tracks
            tracks_dist = new_tracks_dist

        prev = cur
        frame_cnt += 1
    
    # Get Distance matrix
    distMatrix = np.zeros((30, k_max, k_max, 2))
    total_tracks = tracks
    topPoints = list(np.argsort(tracks_dist))

    if (len(topPoints) < k_max):
        nextPoints = list(np.argsort(finished_tracks_dist))
        topPoints += [idx + len(topPoints) for idx in nextPoints]
        total_tracks = tracks + finished_tracks

    if (len(topPoints) > k_max):
        topPoints = topPoints[:k_max]
        total_tracks = total_tracks[:k_max]

    # Find distance between all updated points
    for i in range(len(topPoints)):
        for j in range(len(topPoints)):
            if i == j:
                continue
            for t in range(30):
                if (t > len(total_tracks[i]) - 1) or (t > len(total_tracks[j]) - 1):
                    break
                
                y_diff = total_tracks[i][t][0] - total_tracks[j][t][0]
                x_diff = total_tracks[i][t][1] - total_tracks[j][t][1]
                distMatrix[t,i,j,:] = np.array([x_diff, y_diff])
    
    return distMatrix
    

def main():
    savePath = "GraphData/"
    findPath = "TrimmedData/"
    k_max = 15
    bio_cnt = 0
    non_cnt = 0
    for filename in os.listdir(findPath):
        if filename.endswith(".mov") or filename.endswith(".avi"):
            if "bio" == filename[:3]:
                save_name = savePath + "bio_" + str(bio_cnt) + ".npz"
                graph_1 = getGraphData(findPath + filename, k_max, start_frame = 0)
                with open(save_name, 'wb') as file:
                    np.savez_compressed(file, graph_1)
                bio_cnt += 1 
                save_name = savePath + "bio_" + str(bio_cnt) + ".npz"
                graph_2 = getGraphData(findPath + filename, k_max, start_frame = 30)
                with open(save_name, 'wb') as file:
                    np.savez_compressed(file, graph_2)
                bio_cnt += 1
            else:
                save_name = savePath + "non_" + str(non_cnt) + ".npz"
                graph_1 = getGraphData(findPath + filename, k_max, start_frame = 0)
                with open(save_name, 'wb') as file:
                    np.savez_compressed(file, graph_1)
                non_cnt += 1 
                save_name = savePath + "non_" + str(non_cnt) + ".npz"
                graph_2 = getGraphData(findPath + filename, k_max, start_frame = 30)
                with open(save_name, 'wb') as file:
                    np.savez_compressed(file, graph_2)
                non_cnt += 1




main()