from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
from scipy.stats import zscore

import numpy as np
import cv2

use_cuda = True

class Yolo():
    def __init__(self, cfgfile, weightfile):
        self.bbox_set = list()
        self.m = Darknet(cfgfile)
        print('Loading weights from %s... Done!' % (weightfile))
        print('Detecting Ball and Hoop Position from Yolo...')
        self.m.load_weights(weightfile)

    def detect_cv2(self, num, imgfile):
        #m.print_network()
        if use_cuda:
            self.m.cuda()

        num_classes = self.m.num_classes
        namesfile = 'cfg/ball.names'
        class_names = load_class_names(namesfile)

        (img_h, img_w) = imgfile.shape[:2]
        result = dict()
        #img = cv2.imread(imgfile[i])
        sized = cv2.resize(imgfile, (self.m.width, self.m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
        boxes = do_detect(self.m, sized, 0.4, 0.6, use_cuda)
        boxes[0].sort(key = lambda s: s[4], reverse = True)
        if boxes[0]:
            # print(num)
            for box in boxes[0]:
                x1 = int(box[0]*img_w)
                y1 = int(box[1]*img_h)
                w = int(box[2]*img_w) - x1
                h = int(box[3]*img_h) - y1
                bbox = (x1, y1, w, h)
                result['frame'] = num
                result['bbox'] = bbox
                self.bbox_set.append(result.copy())

            #plot_boxes_cv2(img, boxes[0], savename='predictions' + str(i) + '.jpg', class_names=class_names)

    def clear(self):
        self.bbox_set.clear()

    def run(self, num, frame):
        self.detect_cv2(num, frame)

        return self.bbox_set


def get_trajectory(bbox, fps, w, h, hoop):
    center_h, radius_h = box_to_ball(hoop)
    ball_trajs = [[]] #二維陣列記錄每顆球的各項參數
    ball_latest_frame = [] #記錄每顆球最新的frame
    ball_latest_radius = [] #記錄每顆球最新的半徑
    ball_latest_center = [] #記錄每顆球最新的中心座標
    ball_extreme_pts = [] #記錄每顆球的極點座標 [(Xmax, Xmin, Ymax, Ymin), (...), ...]
    ball_coordinates = [[]] #二維陣列紀錄每條軌跡中球的座標點
    b = dict()
    for i, ball in enumerate(bbox):
        center, r = box_to_ball(ball['bbox'])
        b['frame'] = ball['frame']
        b['center'] = center
        b['radius'] = r
        b['bytracker'] = False #紀錄這顆球是本來就偵測到的 不是後來用tracker補齊的結果
        find_corresponding_ball = False
        if i == 0: #建立第一顆球
            ball_trajs[0].append({'lock': False}) #拿來判斷這顆球是不是沒在動
            ball_trajs[0].append(b.copy())
            ball_latest_frame.append(b['frame'])
            ball_latest_radius.append(r)
            ball_latest_center.append(center)
            ball_extreme_pts.append([center[0], center[0], center[1], center[1]])
            ball_coordinates[0].append(center)
        else:
            for j in range(len(ball_trajs)): #判斷是不是已經存在的球
                #如果球沒在動就鎖住這個軌跡 不能再加新的球
                if len(ball_trajs[j]) == int(fps/10)+1:
                    xmax, xmin, ymax, ymin = ball_extreme_pts[j]
                    if (xmax-xmin) < w*0.01 and (ymax-ymin) < h*0.01:
                        ball_trajs[j][0]['lock'] = True
                if ball_trajs[j][0]['lock']: continue
                #一條軌跡每個frame只能有一個球
                if ball['frame'] == ball_latest_frame[j]: continue
                gap_frame = ball['frame'] - ball_latest_frame[j]
                if abs(r-ball_latest_radius[j])/ball_latest_radius[j] < 0.8: #球的半徑大小差<80%
                    v1 = np.array(list(ball_latest_center[j]))
                    v2 = np.array(list(center))
                    dist = np.sqrt(np.sum(np.square(v1 - v2)))
                    if gap_frame <= int(fps/5) and dist < 1.2 * int(60/fps) * gap_frame * ball_latest_radius[j]: #連續消失的frame數<=0.25秒且距離在frame*1.2r內
                        #判斷為同一顆球
                        find_corresponding_ball = True
                        break
                    v3 = np.array(list(center_h))
                    if np.sqrt(np.sum(np.square(v2 - v3))) < 2*(r+radius_h): #如果球中心和籃框中心的距離<2*(球半徑+框半徑) 開起防止球進籃框detection失效的措施
                        if gap_frame > int(fps/4) and gap_frame < int(fps) and dist < 6 * ball_latest_radius[j]: #就算連續消失超過1/5秒(但小於1秒) 若距離在5*r內就把軌跡接上(希望解決球進籃框時detection失效的問題)
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
                ball_coordinates[j].append(center)
            else: #建立一顆新的球
                ball_trajs.append([])
                x = len(ball_trajs)-1 #得到ball_traj這個list最後的index
                ball_trajs[x].append({'lock': False})
                ball_trajs[x].append(b.copy())
                ball_latest_frame.append(b['frame'])
                ball_latest_radius.append(r)
                ball_latest_center.append(center)
                ball_extreme_pts.append([center[0], center[0], center[1], center[1]])
                ball_coordinates.append([])
                ball_coordinates[x].append(center)

    #過濾掉軌跡中可能存在的離群值
    for i in range(len(ball_coordinates)):
        if ball_trajs[i][0]['lock']: continue
        if len(ball_trajs[i]) < int(fps/6): continue
        c = np.array(ball_coordinates[i])
        z_scores = zscore(c, axis=0)
        # print("before")
        # print(ball_coordinates[i])
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        temp = []
        for x in c[filtered_entries]:
            temp.append(tuple(x))
        ball_coordinates[i] = temp
        # print("after")
        # print(ball_coordinates[i])

    #移除每條軌跡一開始的"鎖"
    for traj in ball_trajs:
        # print(traj[0])
        del traj[0]

    # print("Before")
    # for traj in ball_trajs:
    #     for i in range(len(traj)):
    #         print(traj[i]['frame'], traj[i]['center'])
    #     print("\n")

    #把太短的軌跡(frame總數 < 1/6 秒) and 不動的球(面積小於總面積的0.003)
    #0128 update: 改成座標點的分布會比較好
    rm = []
    for i in range(len(ball_trajs)):
        xmax, xmin, ymax, ymin = ball_extreme_pts[i]
        (x_std, y_std) = np.std(np.array(ball_coordinates[i]), axis=0)
        if len(ball_trajs[i]) < int(fps/6):
            rm.append(i)
        elif (xmax-xmin)*(ymax-ymin)/(w*h) < 0.005: #1203更改 測試2180.mp4
            rm.append(i)
        elif x_std+y_std < 100: # 0202, 畫面大小會影響標準差 所以條件可能要改
            # rm.append(i)
            pass
    j = 0
    for i in range(len(rm)):
        ball_trajs.pop(rm[i]-j)
        ball_latest_frame.pop(rm[i]-j)
        ball_latest_radius.pop(rm[i]-j)
        ball_latest_center.pop(rm[i]-j)
        ball_extreme_pts.pop(rm[i]-j)
        j += 1
    
    # print("After")
    # for traj in ball_trajs:
    #     for i in range(len(traj)):
    #         print(traj[i]['frame'], traj[i]['center'])
    #     print("\n")
    
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


def refine_trajectory(ball_trajs, frame_list, first_frame_num):
    for i in range(len(ball_trajs)):
        ori_len = len(ball_trajs[i])
        clip_offset = ball_trajs[i][0]['frame'] - first_frame_num

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
                tracker.init(frame_list[clip_offset+current_index], box)
                # new = cv2.resize(frame_list[clip_offset+current_index][box[1]:box[1]+box[3], box[0]:box[0]+box[2]], (224, 224))
                # cv2.imwrite('videos/tracker/'+str(current_frame)+'.jpg', new)
                # cv2.waitKey(10)
                for k in range(1, gap):
                    ret, bbox = tracker.update(frame_list[clip_offset+current_index+k])
                    center, r = box_to_ball(bbox)
                    b = dict()
                    b['frame'] = current_frame+k
                    b['center'] = center
                    b['radius'] = r
                    b['bytracker'] = True
                    ball_trajs[i].insert(current_index+k, b)
                offset += (gap-1) #跳過用tracking新加入的frame

    return ball_trajs