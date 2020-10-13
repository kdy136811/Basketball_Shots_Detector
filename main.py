import numpy as np
import os
import cv2
import argparse
import json

import detector
import alphapose
import bei

def get_args():
    parser = argparse.ArgumentParser('Get intput video.')
    parser.add_argument('--video', type=str, default='videos/input/test3.mp4',
                        help='video path', dest='video')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    #get video path
    args = get_args()

    #input video
    filepath = args.video
    outputpath = 'videos/output/balls5.mp4'
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
    print('Detecting Ball Position from Yolo...')
    y = detector.Yolo()
    bbox = y.run(frame_list) #[{'frame': int, 'bbox': (x1, y1, w, h)}, {...}, ......]

    #generate ball(s) trajectory
    print('Generating Ball Trajectories...')
    #hoop = (929, 430, 78, 74)
    #hoop = (951, 409, 61, 57)
    hoop = (933, 431, 73, 76)
    #之後yolo加入籃框辨識後 就會詪bbox一起傳進去 不用另外設
    ball_trajs = detector.get_trajectory(bbox, fps, w, h, hoop) #[[{'frame': int, 'center':(int, int), 'radius': int, 'bytracker': bool},{...},{...}...], [{...},{...}...], ......]

    #refine ball(s) trajectory by tracker CSRT
    print('Refining Ball Trajectories with CSRT tracking algorithm...')
    ball_trajs = detector.refine_trajectory(ball_trajs, frame_list) #[[{'frame': int, 'center':(int, int), 'radius': int, 'bytracker': bool},{...},{...}...], [{...},{...}...], ......]

    #get body skeleton from alphapose
    print('Detecting Body Skeleton from Alphapose...')
    ap = alphapose.AlphaPose(filepath)
    ap.run()
    arms = ap.arm_pos() #[[frame_number, left_x1y1, left_x2y2, right_x1y1, right_x2y2], [...], [...], ......]
    #arms = ap.arm_pos_json('results.json')

    #detect if the balls are released or not
    print('Predicting Ball Release...')
    ball_trajs = detector.release_detector(ball_trajs, arms, len(frame_list))

    #generate basketball energy image
    print('Generating basketball energy image...')
    b = bei.BEI(ball_trajs, hoop, (h,w), fps, filepath.split('/')[2].split('.')[0])
    b.run()
"""
    #illustrate ball's position
    videoWriter = cv2.VideoWriter('./' + outputpath, cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))

    #len(frame_list)
    print('Writing output video!')
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