import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader import VideoLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

import ntpath
import os
import sys
import time
from fn import getTime
import cv2
import json
import time

from pPose_nms import pose_nms, write_json

args = opt
args.dataset = 'coco'
args.sp = True
args.save_video = False
args.outputpath = 'videos/output/'

class AlphaPose():
    def __init__(self, videofile, mode='normal'):
        self.videofile = videofile
        self.mode = mode
        self.data_loader = VideoLoader(self.videofile, batchSize=args.detbatch).start()
        (fourcc,fps,frameSize) = self.data_loader.videoinfo()
        self.fourcc = fourcc
        self.fps = fps
        self.frameSize = frameSize
        self.det_loader = DetectionLoader(self.data_loader, batchSize=args.detbatch).start()
        self.det_processor = DetectionProcessor(self.det_loader).start()
        self.pose_dataset = Mscoco()
        save_path = os.path.join(args.outputpath, 'AlphaPose_'+ntpath.basename(self.videofile).split('.')[0]+'.mp4')
        self.writer = DataWriter(args.save_video, save_path, cv2.VideoWriter_fourcc(*'DIVX'), self.fps, self.frameSize).start()
        self.results = list()
        self.total1 = 0
        self.total2 = 0

    def pose_estimation(self, pose_model):
        batchSize = args.posebatch
        for i in range(self.data_loader.length()):
            with torch.no_grad(): #不計算導數以此減少運算量 可用在model evaluating時 將inference的code放在其中
                t1 = time.time()
                (inps, orig_img, im_name, boxes, scores, pt1, pt2) = self.det_processor.read()
                
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    self.writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                    continue
                t2 = time.time()
                # Pose Estimation
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j*batchSize:min((j + 1)*batchSize, datalen)].cuda()
                    hm_j = pose_model(inps_j)
                    hm.append(hm_j)
                hm = torch.cat(hm)

                hm = hm.cpu().data
                self.writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
                t3 = time.time()
                self.total1 += (t2-t1)
                self.total2 += (t3-t2)


    def run(self):
        args.mode = self.mode
        if args.fast_inference:
            pose_model = InferenNet_fast(4 * 1 + 1, self.pose_dataset)
        else:
            pose_model = InferenNet(4 * 1 + 1, self.pose_dataset)
        pose_model.cuda()
        pose_model.eval()

        #pose estimation
        print('Start Pose Estimating...')
        self.pose_estimation(pose_model)

        print('Finish Model Running.')
        if (args.save_img or args.save_video) and not args.vis_fast:
            print('Rendering remaining images in the queue...')
        while(self.writer.running()):
            pass
        self.writer.stop()
        self.results = self.writer.results().copy()


    def arm_pos(self):
        arms = []
        for frame in self.results:
            #body是每個人的骨架的dictionary
            jpg = frame['imgname'].split('.')
            for body in frame['result']:
                arm = [] #[int, tuple, tuple, tuple, tuple] -> [frame_number, left_x1y1, left_x2y2, right_x1y1, right_x2y2]
                arm.append(int(jpg[0]))
                joint_list = body['keypoints'].tolist()
                arm.append((int(joint_list[7][0]), int(joint_list[7][1])))
                arm.append((int(joint_list[9][0]), int(joint_list[9][1])))
                arm.append((int(joint_list[8][0]), int(joint_list[8][1])))
                arm.append((int(joint_list[10][0]), int(joint_list[10][1])))
                arms.append(arm)

        return arms


    def arm_pos_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        #json_format
        arms = []
        l_index = [21, 27]
        r_index = [24, 30]
        for body in result:
            #body是每個人的骨架的dictionary
            arm = [] #[int, tuple, tuple, tuple, tuple] -> [frame_number, left_x1y1, left_x2y2, right_x1y1, right_x2y2]
            jpg = body['image_id'].split('.')
            arm.append(int(jpg[0]))
            for i in l_index:
                arm.append((int(body['keypoints'][i]), int(body['keypoints'][i+1])))
            for i in r_index:
                arm.append((int(body['keypoints'][i]), int(body['keypoints'][i+1])))
            arms.append(arm)

        return arms
