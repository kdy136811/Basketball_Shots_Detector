# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet

import numpy as np
import cv2

"""hyper parameters"""
use_cuda = True

class Yolo():
    def __init__(self):
        self.bbox_set = list()


    def detect_cv2(self, cfgfile, weightfile, imgfile):
        import cv2
        m = Darknet(cfgfile)

        m.print_network()
        m.load_weights(weightfile)
        print('Loading weights from %s... Done!' % (weightfile))

        if use_cuda:
            m.cuda()

        num_classes = m.num_classes
        namesfile = 'cfg/ball.names'
        class_names = load_class_names(namesfile)

        (img_h, img_w) = imgfile[0].shape[:2]
        result = dict()
        for i, img in enumerate(imgfile):
            #img = cv2.imread(imgfile[i])
            sized = cv2.resize(img, (m.width, m.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        
            boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
            boxes[0].sort(key = lambda s: s[4], reverse = True)
            if boxes[0]:
                for box in boxes[0]:
                    x1 = int(box[0]*img_w)
                    y1 = int(box[1]*img_h)
                    w = int(box[2]*img_w) - x1
                    h = int(box[3]*img_h) - y1
                    bbox = (x1, y1, w, h)
                    result['frame'] = i
                    result['bbox'] = bbox
                    self.bbox_set.append(result.copy())

            #plot_boxes_cv2(img, boxes[0], savename='predictions' + str(i) + '.jpg', class_names=class_names)


    def run(self, frames):
        cfg = 'cfg/yolo-ball.cfg'
        weights = 'cfg/yolo-ball_final.weights'
        self.detect_cv2(cfg, weights, frames)

        return self.bbox_set


def get_trajectory(bbox, fps, w, h):
    ball_trajs = [[]] #二維陣列記錄每顆球的軌跡
    ball_latest_frame = [] #記錄每顆球最新的frame
    ball_latest_radius = [] #記錄每顆球最新的半徑
    ball_latest_center = [] #記錄每顆球最新的中心座標
    ball_extreme_pts = [] #記錄每顆球的極點座標 [(Xmax, Xmin, Ymax, Ymin), (...), ...]
    b = dict()
    for i, ball in enumerate(bbox):
        center, r = box_to_ball(ball['bbox'])
        b['frame'] = ball['frame']
        b['center'] = center
        b['radius'] = r
        b['bytracker'] = False #紀錄這顆球是本來就偵測到的 不是後來用tracker補齊的結果
        find_corresponding_ball = False
        if i == 0: #建立第一顆球
            ball_trajs[0].append(b.copy())
            ball_latest_frame.append(b['frame'])
            ball_latest_radius.append(r)
            ball_latest_center.append(center)
            ball_extreme_pts.append([center[0], center[0], center[1], center[1]])
        else:
            for j in range(len(ball_trajs)): #判斷是不是已經存在的球
                if ball['frame'] == ball_latest_frame[j]: continue
                gap_frame = ball['frame'] - ball_latest_frame[j]
                if abs(r-ball_latest_radius[j])/ball_latest_radius[j] < 0.8: #球的半徑大小差<80%
                    v1 = np.array(list(ball_latest_center[j]))
                    v2 = np.array(list(center))
                    dist = np.sqrt(np.sum(np.square(v1 - v2)))
                    if gap_frame <= int(fps/5) and dist < 1.2 * gap_frame * ball_latest_radius[j]: #連續消失的frame數<=12且距離在frame*1.2r內
                        #判斷為同一顆球
                        find_corresponding_ball = True
                        break
                    # 10/10要來加上和籃框位置接近與否的判斷!!
                    elif gap_frame > 10 and gap_frame < int(fps) and dist < 5 * ball_latest_radius[j]: #就算連續消失超過1/5秒(但小於1秒) 若距離在5*r內就把軌跡接上(希望解決球進籃框時detection失效的問題)
                        #判斷為同一顆球
                        find_corresponding_ball = True
                        break
                
            if find_corresponding_ball:
                ball_trajs[j].append(b.copy())
                ball_latest_frame[j] = b['frame']
                ball_latest_radius[j] = r
                ball_latest_center[j] = center
                if center[0] > ball_extreme_pts[j][0]: ball_extreme_pts[j][0] = center[0]
                if center[0] < ball_extreme_pts[j][1]: ball_extreme_pts[j][1] = center[0]
                if center[1] > ball_extreme_pts[j][2]: ball_extreme_pts[j][2] = center[1]
                if center[1] < ball_extreme_pts[j][3]: ball_extreme_pts[j][3] = center[1]
            else: #建立一顆新的球
                ball_trajs.append([])
                x = len(ball_trajs)-1 #得到ball_traj這個list最後的index
                ball_trajs[x].append(b.copy())
                ball_latest_frame.append(b['frame'])
                ball_latest_radius.append(r)
                ball_latest_center.append(center)
                ball_extreme_pts.append([center[0], center[0], center[1], center[1]])
    #把太短的軌跡(frame總數 < 10) and 不動的球(面積小於總面積的0.003)
    rm = []
    for i in range(len(ball_trajs)):
        xmax, xmin, ymax, ymin = ball_extreme_pts[i]
        if len(ball_trajs[i]) < 10:
            rm.append(i)
        elif (xmax-xmin)*(ymax-ymin)/(w*h) < 0.003:
            rm.append(i)
    j = 0
    for i in range(len(rm)):
        ball_trajs.pop(rm[i]-j)
        ball_latest_frame.pop(rm[i]-j)
        ball_latest_radius.pop(rm[i]-j)
        ball_latest_center.pop(rm[i]-j)
        ball_extreme_pts.pop(rm[i]-j)
        j += 1

    return ball_trajs


def box_to_ball(box):
    x, y, w, h = box
    center_x = int(x+(w/2))
    center_y = int(y+(h/2))
    r = int(max(w, h)/2)

    return (center_x, center_y), r

def ball_to_box(center, r):
    center_x, center_y = center
    x = int(center_x-r)
    y = int(center_y-r)

    return (x, y, 2*r, 2*r)


def refine_trajectory(ball_trajs, frame_list):
    for i in range(len(ball_trajs)):
        ori_len = len(ball_trajs[i])
        offset = 0
        for j in range(ori_len):
            current_index = j+offset
            if current_index == len(ball_trajs[i])-1: break
            current_frame = ball_trajs[i][current_index]['frame']
            next_frame = ball_trajs[i][current_index+1]['frame']
            gap = next_frame - current_frame
            if gap > 1: #如果有斷掉的frame用tracker補起來
                tracker = cv2.TrackerCSRT_create()
                box = ball_to_box(ball_trajs[i][current_index]['center'], ball_trajs[i][current_index]['radius'])
                tracker.init(frame_list[current_frame], box)
                for k in range(1, gap):
                    ret, bbox = tracker.update(frame_list[current_frame+k])
                    center, r = box_to_ball(bbox)
                    b = dict()
                    b['frame'] = current_frame+k
                    b['center'] = center
                    b['radius'] = r
                    b['bytracker'] = True
                    ball_trajs[i].insert(current_index+k, b)
                offset += (gap-1) #跳過用tracking新加入的frame

    return ball_trajs  


def released_detector(ball_trajs, arms, frame_num):
    index_arm = 0
    index_ball = [0]*len(ball_trajs)
    for i in range(frame_num):
        wrist = []
        #找出每個frame中手腕的座標點
        while arms[index_arm][0] == i:
            wrist.append([arms[index_arm][2][0], arms[index_arm][2][1]])
            wrist.append([arms[index_arm][4][0], arms[index_arm][4][1]])
            if index_arm == len(arms)-1:
                break
            index_arm += 1
        #和同個frame偵測到的每顆球的中心位置做距離判斷
        for j in range(len(ball_trajs)):
            if index_ball[j] == len(ball_trajs[j]): continue
            if ball_trajs[j][index_ball[j]]['frame'] == i:
                ball_trajs[j][index_ball[j]]['release'] = True
                v1 = np.array(list(ball_trajs[j][index_ball[j]]['center']))
                for k in range(len(wrist)):
                    v2 = np.array(wrist[k])
                    if np.sqrt(np.sum(np.square(v1 - v2))) < (2*ball_trajs[j][index_ball[j]]['radius']): #如果求座標跟任何一個手腕座標相距<球直徑
                        ball_trajs[j][index_ball[j]]['release'] = False
                        break
                index_ball[j] += 1

    return ball_trajs
          