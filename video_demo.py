# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 20:27:13 2020

@author: Leacius
"""

import cv2
from random import randint
from PIL import Image

if __name__ == '__main__':
    vid = cv2.VideoCapture("videos/input/test1.mp4")
    
    success, frame = vid.read()
    bbox = cv2.selectROI(frame, False)
    cv2.destroyAllWindows()
    print(bbox)
"""
    tracker = cv2.TrackerCSRT_create()
    multiTracker = cv2.MultiTracker_create()
    multiTracker.add(tracker, frame, bbox)
    
    r,c,z = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('track1.mp4', fourcc, 60, (frame.shape[1], frame.shape[0]))
    colors = ((randint(0, 255), randint(0, 255), randint(0, 255)))
    
    # img = []
    i = 0
    if vid.isOpened():
        while True:
            success, frame = vid.read()
            if success == False:
                
                break
            success, boxes = multiTracker.update(frame)
            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, colors, 2, 1)
            # ims = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
            # img.append(ims)
            cv2.imshow('video', frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            i = i+1
    # img[0].save('C:/Users/Leacius/Desktop/cc.gif',
    #                            save_all=True, append_img=img[1:], optimize=False, duration=40, loop=0)
    vid.release()
    cv2.destroyAllWindows()
"""