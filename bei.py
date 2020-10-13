import cv2
import os
import numpy as np

class BEI():
    def __init__(self, ball_trajs, hoop, size, fps=60, fname='vid'):
        self.ball_trajs = ball_trajs
        self.hoop = hoop
        self.clips = list()
        self.size = size
        self.fps = fps
        self.fname = fname
        self.bei_size = (128, 128) #64以上為32的倍數


    def generate_bei(self):
        filepath = 'videos/bei/'+ str(self.fname)
        if not (os.path.isdir(filepath)):
            os.mkdir(filepath)
        for i in range(len(self.clips)):
            key_frame = self.clips[i][0][1]
            img = np.zeros(self.size, np.uint8) #gray
            #img = np.zeros((self.size[0], self.size[1], 3), np.uint8) #rgb
            T = len(self.clips[i])-1 #減掉[0]拿來存keyframe的tuple
            for j in range(1, len(self.clips[i])):
                x, y = self.clips[i][j]['center']
                r = self.clips[i][j]['radius']
                if j == 1:
                    ball_extreme_pts = [x+r, x-r, y+r, y-r] #[Xmax, Xmin, Ymax, Ymin]
                else:
                    if x+r > ball_extreme_pts[0]: ball_extreme_pts[0] = x+r
                    if x-r < ball_extreme_pts[1]: ball_extreme_pts[1] = x-r
                    if y+r > ball_extreme_pts[2]: ball_extreme_pts[2] = y+r
                    if y-r < ball_extreme_pts[3]: ball_extreme_pts[3] = y-r
                cv2.circle(img, self.clips[i][j]['center'], self.clips[i][j]['radius'], int(255*(j/T)), -1) #gray
                #cv2.circle(img, self.clips[i][j]['center'], self.clips[i][j]['radius'], (int(255*(j/T)), int(255*(j/T)), int(255*(j/T))), -1) #rgb
            #畫籃框
            #cv2.rectangle(img, (self.hoop[0], self.hoop[1]), (self.hoop[0]+self.hoop[2], int(self.hoop[1]+self.hoop[3]/3)), (0, 0, 255), -1)
            
            #考慮籃框位置是否超過極點
            center_h, radius_h = self.box_to_ball(self.hoop)
            if center_h[0]+radius_h > ball_extreme_pts[0]: ball_extreme_pts[0] = center_h[0]+radius_h
            if center_h[0]-radius_h < ball_extreme_pts[1]: ball_extreme_pts[1] = center_h[0]-radius_h
            if center_h[1]+radius_h > ball_extreme_pts[2]: ball_extreme_pts[2] = center_h[1]+radius_h
            if center_h[1]-radius_h < ball_extreme_pts[3]: ball_extreme_pts[3] = center_h[1]-radius_h

            #crop and resize
            crop_h = ball_extreme_pts[2]-ball_extreme_pts[3]
            crop_w = ball_extreme_pts[0]-ball_extreme_pts[1]
            crop = img[ball_extreme_pts[3]:ball_extreme_pts[2], ball_extreme_pts[1]:ball_extreme_pts[0]]
            rst = np.zeros(self.bei_size, np.uint8)
            #長邊resize為基準
            if max(crop_h, crop_w) == crop_h: #crop_h大
                ratio = crop_h/(self.bei_size[0]/16*15)
                rsz_h = int(self.bei_size[0]/16*15)
                rsz_w = int(crop_w/ratio)
                start_h = int(self.bei_size[0]/32)
                start_w = int((self.bei_size[0]-rsz_w)/2)
                rst[start_h : start_h+rsz_h, start_w : start_w+rsz_w] = cv2.resize(crop, (rsz_w, rsz_h))
            else: #crop_w大
                ratio = crop_w/(self.bei_size[0]/16*15)
                rsz_w = int(self.bei_size[0]/16*15)
                rsz_h = int(crop_h/ratio)
                start_w = int(self.bei_size[0]/32)
                start_h = int((self.bei_size[0]-rsz_h)/2)
                rst[start_h : start_h+rsz_h, start_w : start_w+rsz_w] = cv2.resize(crop, (rsz_w, rsz_h))

            cv2.imwrite(filepath + '/rsz_' + str(key_frame) + '.jpg', rst)
            #cv2.imwrite(filepath + '/' + str(key_frame) + '.jpg', img)


    def select_clip(self):
        center_h, radius_h = self.box_to_ball(self.hoop)
        v1 = np.array(list(center_h)) #籃框位置
        for i in range(len(self.ball_trajs)):
            offset = 0
            for j in range(len(self.ball_trajs[i])):
                if j+offset >= len(self.ball_trajs[i]): break
                v2 = np.array(list(self.ball_trajs[i][j+offset]['center'])) #key_frame時球的位置
                #球半徑<框半徑 and 球中心與框中心距離<1.5*(兩半徑相加) and 球離手 則紀錄為keyframe
                if self.ball_trajs[i][j+offset]['radius'] < radius_h and np.sqrt(np.sum(np.square(v1 - v2))) < 1.5*(radius_h + self.ball_trajs[i][j+offset]['radius']) and self.ball_trajs[i][j+offset]['release']:
                    clip = []
                    temp = {}
                    temp['center'] = self.ball_trajs[i][j+offset]['center']
                    temp['radius'] = self.ball_trajs[i][j+offset]['radius']
                    clip.append(temp.copy())
                    end_pre = False
                    end_lat = False
                    first = 0
                    last = 0
                    for k in range(1, int(self.fps)):
                        if j+offset-k >= 0 and self.ball_trajs[i][j+offset-k]['release'] and not end_pre: #往前insert軌跡(index沒<0 and 球是離手狀態)
                            temp['center'] = self.ball_trajs[i][j+offset-k]['center']
                            temp['radius'] = self.ball_trajs[i][j+offset-k]['radius']
                            clip.insert(0, temp.copy())
                            first = k
                        else:
                            end_pre = True
                        if j+offset+k < len(self.ball_trajs[i]) and self.ball_trajs[i][j+offset+k]['release'] and not end_lat: #往後append軌跡(index沒>len and 球是離手狀態)
                            temp['center'] = self.ball_trajs[i][j+offset+k]['center']
                            temp['radius'] = self.ball_trajs[i][j+offset+k]['radius']
                            clip.append(temp.copy())
                            last = k
                            if last == int(self.fps)-1:
                                #如果key_frame後超過一秒球還是在籃框附近 則繼續記錄軌跡
                                #框與球距離<2.5*(兩半徑相加) and 球離手
                                last += 1 #測試新的frame
                                while np.sqrt(np.sum(np.square(v1 - np.array(list(self.ball_trajs[i][j+offset+last]['center']))))) < 2.5*(radius_h + self.ball_trajs[i][j+offset+last]['radius']) and self.ball_trajs[i][j+offset+last]['release']:
                                    temp['center'] = self.ball_trajs[i][j+offset+last]['center']
                                    temp['radius'] = self.ball_trajs[i][j+offset+last]['radius']
                                    clip.append(temp.copy())
                                    last += 1
                                last -= 1 #把最後一次迴圈多加的1扣回來
                        else:
                            end_lat = True
                        if end_pre and end_lat:
                            break
                    key_frame = self.ball_trajs[i][j+offset]['frame']
                    clip.insert(0, (key_frame-first, key_frame, key_frame+last)) #紀錄擷取出的畫面的frame範圍
                    if last > int(self.fps/5): #確保球在靠近籃框後有足夠的資訊
                        self.clips.append(clip)
                    offset += last #跳到這個clip後面


    def box_to_ball(self, box):
        x, y, w, h = box
        center_x = int(x+(w/2))
        center_y = int(y+(h/2))
        r = int(max(w, h)/2)

        return (center_x, center_y), r


    def run(self):
        if not (os.path.isdir('videos/bei')):
            os.mkdir('videos/bei')
        self.select_clip()
        self.generate_bei()