import numpy as np
import os
import cv2
import argparse
import json
import time

import detector
import bei

def box_to_ball(box):
        x, y, w, h = box
        center_x = int(x+(w/2))
        center_y = int(y+(h/2))
        r = int(max(w, h)/2)

        return (center_x, center_y), r

def pipeline(traj, fps, w, h, hoop, frame_list, crop_list, cnt, first_frame_num, crop_size, crop_para):
    _, r_h = box_to_ball(hoop)
    crop_h, crop_w = crop_list[0].shape[:2]

    print('Generating Ball Trajectories...')
    ball_trajs = detector.get_trajectory(traj, fps, r_h*10, r_h*10, hoop)

    print('Refining Ball Trajectories with CSRT tracking algorithm...')
    ball_trajs = detector.refine_trajectory(ball_trajs, crop_list, first_frame_num) #[[{'frame': int, 'center':(int, int), 'radius': int, 'bytracker': bool},{...},{...}...], [{...},{...}...], ......]
    # print(ball_trajs)

    if not ball_trajs: return

    print('Generating Basketball Energy Image...')
    bei.generate_bei(ball_trajs, crop_size, crop_para, filepath.split('/')[len(filepath.split('/'))-1].split('.')[0])

    videoWriter = cv2.VideoWriter('./' + outputpath + str(cnt) + '.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, (crop_w, crop_h))

    #len(frame_list)
    print('Writing output video!')
    index_ball = [0]*len(ball_trajs)
    for i in range(len(crop_list)):
        #draw balls
        for j in range(len(ball_trajs)):
            if index_ball[j] == len(ball_trajs[j]): continue
            if ball_trajs[j][index_ball[j]]['frame']-first_frame_num == i:
                cv2.circle(crop_list[i], ball_trajs[j][index_ball[j]]['center'], ball_trajs[j][index_ball[j]]['radius'], (0, 255, 0), 3)
                
                index_ball[j] += 1

        videoWriter.write(crop_list[i])
    
    videoWriter.release()
    
    print('Finished the Clip!! :)')


if __name__ == '__main__':
    mypath = 'videos/input/'
    files = os.listdir(mypath)
    for f in files:
        filepath = mypath + f
        print(filepath)
        
        outputpath = 'videos/output/' + filepath.split("/")[2].split('.')[0] + '_'
        print('Extracting Video Frames...')

        vc = cv2.VideoCapture(filepath)
        fourcc = int(vc.get(cv2.CAP_PROP_FOURCC))
        fps = int(vc.get(cv2.CAP_PROP_FPS))
        # frameSize = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        ret, frame = vc.read()
        h, w = frame.shape[:2]
        print(h, w)

        if h > 1080:
            m = h/1080
            frame = cv2.resize(frame, (int(w/m), int(h/m)))
        hoop = cv2.selectROI(frame, False)
        cv2.destroyAllWindows()
        
        t1 = time.time()

        center_h, radius_h = box_to_ball(hoop)
        print(center_h, radius_h)

        crop_para = 7
        crop_size = 800

        cnt = 0
        cnt_no_ball = 0
        traj = []
        frame_list = []
        crop_list = []
        clip_thresh = int(fps/5)
        y = detector.Yolo('cfg/yolo-ball.cfg', 'cfg/yolo-ball_last.weights')
        while ret:
            if h > 1080:
                m = h/1080
                frame = cv2.resize(frame, (int(w/m), int(h/m)))

            frame_list.append(frame)
            # crop the area near hoop
            frame = frame[center_h[1]-crop_para*radius_h:center_h[1]+crop_para*radius_h, center_h[0]-crop_para*radius_h:center_h[0]+crop_para*radius_h]
            # cv2.imshow('f', frame)
            # cv2.waitKey(5)
            frame = cv2.resize(frame, (800, 800), interpolation=cv2.INTER_LINEAR)
            crop_list.append(frame)

            # run yolo
            bbox = y.run(cnt, frame)
            if not bbox:
                # print("nothing", cnt_no_ball)
                cnt_no_ball += 1
                if cnt_no_ball >= clip_thresh:
                    # end the clip
                    cnt_no_ball = 0
                    if traj:
                        clip_len = len(crop_list)
                        if len(traj) > clip_thresh:
                            traj_len = traj[len(traj)-1]['frame'] - traj[0]['frame'] + 1
                            # print(traj_len)
                            # print("clip", clip_len, cnt-clip_thresh)
                            # print("final", clip_len-clip_thresh, clip_len-traj_len)
                            end = clip_len - clip_thresh
                            start = end - traj_len
                            first_frame_num = traj[0]['frame']
                            pipeline(traj, fps, w, h, hoop, frame_list[start:end], crop_list[start:end], cnt-clip_thresh, first_frame_num, crop_size, crop_para)
                        traj.clear()
                        frame_list.clear()
                        crop_list.clear()
                    else:
                        ret, frame = vc.read()
                        frame_list.clear()
                        crop_list.clear()
                        continue
            else:
                cnt_no_ball = 0
            
            traj += bbox
            cnt += 1

            y.clear()
            ret, frame = vc.read()

        if traj:
            clip_len = len(crop_list)
            if len(traj) > clip_thresh:
                traj_len = traj[len(traj)-1]['frame'] - traj[0]['frame'] + 1
                # a little bit different here
                end = clip_len - cnt_no_ball
                start = end - traj_len
                first_frame_num = traj[0]['frame']
                pipeline(traj, fps, w, h, hoop, frame_list[start:end], crop_list[start:end], cnt-clip_thresh, first_frame_num, crop_size, crop_para)
            traj.clear()
            frame_list.clear()
            crop_list.clear()

        t2 = time.time()
        print('Finished Everthing. Congrats!! :)')
        print('TOTAL TIME', round(t2-t1, 3))

        vc.release()