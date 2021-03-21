import cv2
import os
import numpy as np

def generate_bei(ball_trajs, crop_size, crop_para, fname='vid'):
    filepath = 'videos/bei/'+ fname + '/'
    if not (os.path.isdir(filepath)):
        os.mkdir(filepath)
    for traj in ball_trajs:
        img = np.zeros((crop_size, crop_size, 3), np.uint8) #rgb
        T = traj[len(traj)-1]['frame'] - traj[0]['frame']
        # draw hoop
        r = int(crop_size/2/crop_para)
        xy = int(crop_size/2 - r)
        cv2.rectangle(img, (xy, xy), (xy+2*r, int(xy + 2*r/3)), (0, 0, 255), -1)
        for j in range(len(traj)):
            # draw ball
            cv2.circle(img, traj[j]['center'], traj[j]['radius'], (int(255*(j/T)), int(255*(j/T)), int(255*(j/T))), -1) #rgb

        rsz = 224
        img = cv2.resize(img, (rsz, rsz))
        cv2.imwrite(filepath + str(traj[0]['frame']) + '.jpg', img)
