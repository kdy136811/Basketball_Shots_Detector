import numpy as np
import os
import cv2
import argparse
import json

import detector
import alphapose

def get_args():
    parser = argparse.ArgumentParser('Get intput video.')
    parser.add_argument('--video', type=str, default='videos/input/test.mp4',
                        help='video path', dest='video')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    #get video path
    args = get_args()

    #input video
    filepath = args.video
    outputpath = 'videos/output/balls1-4.mp4'
    print('Parsing frames from video...')
    vc = cv2.VideoCapture(filepath)
    fps = vc.get(cv2.CAP_PROP_FPS)
    frame_list = list()
    ret, frame = vc.read()
    #i = 10
    while ret:
        #i -= 1
        frame_list.append(frame)
        ret, frame = vc.read()
    h, w = frame_list[0].shape[:2]

    #get bounding boxes from yolo
    print('Getting Ball Position from Yolo...')
    y = detector.Yolo()
    bbox = y.run(frame_list) #[{'frame': int, 'bbox': (x1, y1, w, h)}, {...}, ......]

    #generate ball(s) trajectory
    print('Generate Ball Trajectories...')
    ball_trajs = detector.get_trajectory(bbox, fps, w, h) #[[{'frame': int, 'center':(int, int), 'radius': int, 'bytracker': bool},{...},{...}...], [{...},{...}...], ......]

    #refine ball(s) trajectory by tracker CSRT
    print('Refine Ball Trajectories with CSRT tracking algorithm...')
    ball_trajs = detector.refine_trajectory(ball_trajs, frame_list) #[[{'frame': int, 'center':(int, int), 'radius': int, 'bytracker': bool},{...},{...}...], [{...},{...}...], ......]

    #get body skeleton from alphapose
    print('Getting Body Skeleton from Alphapose...')
    ap = alphapose.AlphaPose(filepath)
    #ap.run()
    #arms = ap.arm_pos() #[[frame_number, left_x1y1, left_x2y2, right_x1y1, right_x2y2], [...], [...], ......]
    arms = ap.arm_pos_json('results.json')

    #detect if the balls are released or not
    print('Ball Release Predicting...')
    ball_trajs = detector.released_detector(ball_trajs, arms, len(frame_list))

    #illustrate ball's position
    videoWriter = cv2.VideoWriter('./' + outputpath, cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))

    #len(frame_list)
    print('Start drawing image!')
    index_arm = 0
    index_ball = [0]*len(ball_trajs)
    for i in range(len(frame_list)):
        #draw arms
        while arms[index_arm][0] == i:
            cv2.line(frame_list[i], arms[index_arm][1], arms[index_arm][2], (0, 255, 255), 3)
            cv2.line(frame_list[i], arms[index_arm][3], arms[index_arm][4], (255, 0, 0), 3)
            if index_arm == len(arms)-1:
                break
            index_arm += 1
        #draw balls
        for j in range(len(ball_trajs)):
            if index_ball[j] == len(ball_trajs[j]): continue
            if ball_trajs[j][index_ball[j]]['frame'] == i:
                if ball_trajs[j][index_ball[j]]['release']:
                    cv2.circle(frame_list[i], ball_trajs[j][index_ball[j]]['center'], ball_trajs[j][index_ball[j]]['radius'], (0, 255, 0), 3)
                else:
                    cv2.circle(frame_list[i], ball_trajs[j][index_ball[j]]['center'], ball_trajs[j][index_ball[j]]['radius'], (0, 0, 255), 3)
                index_ball[j] += 1

        videoWriter.write(frame_list[i])
    print('Finished Everthing. Congrats!! :)')
    vc.release()
    videoWriter.release()

"""    
    for i in range(len(frame_list)):
        wrist = []
        #draw arms
        while arms[index][0] == i:
            cv2.line(frame_list[i], arms[index][1], arms[index][2], (0, 255, 255), 3)
            cv2.line(frame_list[i], arms[index][3], arms[index][4], (255, 0, 0), 3)
            wrist.append([arms[index][2][0], arms[index][2][1]])
            wrist.append([arms[index][4][0], arms[index][4][1]])
            if index == len(arms)-1:
                break
            index += 1
        #draw balls
        while bbox[index_box]['frame'] == i:
            x1 = bbox[index_box]['bbox'][0]
            y1 = bbox[index_box]['bbox'][1]
            w = bbox[index_box]['bbox'][2]
            h = bbox[index_box]['bbox'][3]
            x_center = int(x1+(w/2))
            y_center = int(y1+(h/2))
            r = int(max(w, h)/2)
            #判定球是否離手
            off_hand = True
            v1 = np.array([x_center, y_center])
            for j in range(len(wrist)):
                v2 = np.array(wrist[j])
                if np.sqrt(np.sum(np.square(v1 - v2))) < (2*r):
                    off_hand = False
                    break
            if off_hand:
                cv2.circle(frame_list[i], (x_center, y_center), r, (0, 255, 0), 3)
            else:
                cv2.circle(frame_list[i], (x_center, y_center), r, (0, 0, 255), 3)
            if index_box == len(bbox)-1:
                break
            index_box += 1

        videoWriter.write(frame_list[i])
    print('Finished Everthing. Congrats!! :)')
    vc.release()
    videoWriter.release()
"""