import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import random
import torchvision
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import copy
import os
import traceback
import datetime
from torchvision import datasets, transforms, models
from efficientnet_pytorch import EfficientNet
import math
import time
import cv2
import sys

dir_path = os.path.dirname(__file__)
model_b0_path = dir_path + '/PretrainedModels/Ours/ef_b0.pth'
model_name = 'efficientnet-b0'
center_face_path = dir_path + '/CenterFace-master/models/onnx/centerface.onnx'

class CenterFace(object):
    def __init__(self, landmarks=True):
        self.landmarks = landmarks        
        self.net = cv2.dnn.readNetFromONNX(center_face_path)        
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = 0, 0, 0, 0

    def __call__(self, img, height, width, threshold=0.5):
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = self.transform(height, width)
        return self.inference_opencv(img, threshold)

    def inference_opencv(self, img, threshold):
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(self.img_w_new, self.img_h_new), mean=(0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        begin = datetime.datetime.now()
        if self.landmarks:
            heatmap, scale, offset, lms = self.net.forward(["537", "538", "539", '540'])
        else:
            heatmap, scale, offset = self.net.forward(["535", "536", "537"])
        end = datetime.datetime.now()
        print("cpu times = ", end - begin)
        return self.postprocess(heatmap, lms, offset, scale, threshold)

    def transform(self, h, w):
        img_h_new, img_w_new = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        scale_h, scale_w = img_h_new / h, img_w_new / w
        return img_h_new, img_w_new, scale_h, scale_w

    def postprocess(self, heatmap, lms, offset, scale, threshold):
        if self.landmarks:
            dets, lms = self.decode(heatmap, scale, offset, lms, (self.img_h_new, self.img_w_new), threshold=threshold)
        else:
            dets = self.decode(heatmap, scale, offset, None, (self.img_h_new, self.img_w_new), threshold=threshold)
        if len(dets) > 0:
            dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2] / self.scale_w, dets[:, 1:4:2] / self.scale_h
            if self.landmarks:
                lms[:, 0:10:2], lms[:, 1:10:2] = lms[:, 0:10:2] / self.scale_w, lms[:, 1:10:2] / self.scale_h
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            if self.landmarks:
                lms = np.empty(shape=[0, 10], dtype=np.float32)
        if self.landmarks:
            return dets, lms
        else:
            return dets

    def decode(self, heatmap, scale, offset, landmark, size, threshold=0.1):
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        if self.landmarks:
            boxes, lms = [], []
        else:
            boxes = []
        if len(c0) > 0:
            for i in range(len(c0)):
                s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
                x1, y1 = min(x1, size[1]), min(y1, size[0])
                boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
                if self.landmarks:
                    lm = []
                    for j in range(5):
                        lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                        lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
                    lms.append(lm)
            boxes = np.asarray(boxes, dtype=np.float32)
            keep = self.nms(boxes[:, :4], boxes[:, 4], 0.3)
            boxes = boxes[keep, :]
            if self.landmarks:
                lms = np.asarray(lms, dtype=np.float32)
                lms = lms[keep, :]
        if self.landmarks:
            return boxes, lms
        else:
            return boxes

    def nms(self, boxes, scores, nms_thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]
        num_detections = boxes.shape[0]
        suppressed = np.zeros((num_detections,), dtype=np.bool)

        keep = []
        for _i in range(num_detections):
            i = order[_i]
            if suppressed[i]:
                continue
            keep.append(i)

            ix1 = x1[i]
            iy1 = y1[i]
            ix2 = x2[i]
            iy2 = y2[i]
            iarea = areas[i]

            for _j in range(_i + 1, num_detections):
                j = order[_j]
                if suppressed[j]:
                    continue

                xx1 = max(ix1, x1[j])
                yy1 = max(iy1, y1[j])
                xx2 = min(ix2, x2[j])
                yy2 = min(iy2, y2[j])
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                inter = w * h
                ovr = inter / (iarea + areas[j] - inter)
                if ovr >= nms_thresh:
                    suppressed[j] = True

        return keep

class MyModel(nn.Module):
    def __init__(self, model_name: str):
        super(MyModel,self).__init__()

        # self.model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
        self.model = EfficientNet.from_pretrained(model_name)
        self.model.set_swish(memory_efficient=False)
        self.linear1 = torch.nn.Linear(1000, 1)
        self.activation3 = torch.nn.Sigmoid()

    def forward(self,x):
        answ = self.model(x)
        #print(answ.shape)
        #answ = self.encoder(answ)
        #print(answ.shape)
        answ = self.activation3(self.linear1(answ))

        return answ

def detect_faces(image):    
    final_boxes = []
    h, w = image.shape[:2]
    landmarks = True
    centerface = CenterFace(landmarks=landmarks)
    if landmarks:
        dets, lms = centerface(image, h, w, threshold=0.35)
    else:
        dets = centerface(image, threshold=0.35)

    for det in dets:
        boxes, score = det[:4], det[4]
        w = abs(int(boxes[0])-int(boxes[2])) 
        h = abs(int(boxes[1])-int(boxes[3]))
        if (True):            
            final_boxes.append(boxes)    
    return final_boxes

transform_efnet=transforms.Compose([transforms.Resize((150,150)),
                              transforms.ToTensor()
                              ])

def init_ef_model(model_name, model_path):
    device = torch.device('cpu')
    model = MyModel(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval();
    return model

def get_res_efnet(image_path: str, model: MyModel):
    image = Image.open(image_path)
    image = transform_efnet(image).unsqueeze(0)

    # model.cuda()
    # Set layers such as dropout and batchnorm in evaluation mode
    # model.eval();

    # Get the 1000-dimensional model output
    with torch.no_grad():
        out = model(image)
    print(out.detach().numpy()[0][0])    
    return out.detach().numpy()[0][0]

def predict_gender(image):
    b0_model = init_ef_model(model_name, model_b0_path)   
    im_pil = Image.fromarray(image)
    image_data = transform_efnet(im_pil).unsqueeze(0)    

    # Get the 1000-dimensional model output
    with torch.no_grad():
        out = b0_model(image_data)

    out = round(out.detach().numpy()[0][0], 3)        
    gender = 'Male' if out > 0.5 else 'Female'
    confidence = out if out > 0.5 else (1-out)
    return gender, confidence

def analyze_image(image):
    predictions = [] # [[x, y, w, h, gender, confidence],...]              
    boxes = detect_faces(image)    
    print(f'{len(boxes)} faces detected using CenterFace.')
    height, width = image.shape[:2]
    for (x, y, w, h) in boxes:   
        treshold = max(abs(x-w), abs(y-h))
        x1 = max(round(x - treshold * 0.5), 0)
        y1 = max(round(y - treshold * 0.5), 0)
        x2 = min(round(w + treshold * 0.5), width)        
        y2 = min(round(h + treshold * 0.5), height)               
        crop_face = image[y1:y2, x1:x2]        
        
        gender, confidence = predict_gender(crop_face)
        
        predictions.append([x, y, w, h, gender, round(confidence, 3)])
        print(gender)            
    return predictions

def draw_res(image, predictions):    

    font                   = cv2.FONT_HERSHEY_SIMPLEX        
    fontColor              = (255,255,255)
    lineType               = 1
    thickness              = 1
    height, width = image.shape[:2]
    for (x1, y1, x2, y2, gender, conf) in predictions:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (2, 255, 0), 1)
        bottomLeftCornerOfText = (int(x1), int(y1))
        fontScale = min(width,height)/(2500)
        cv2.putText(image, str(gender), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)

    return image


def __main__(argv):    
    parser = argparse.ArgumentParser(description='Gender estimation')    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        exit(1)
    
    parser.add_argument('-i', '--input_path', type=str, help='Input image path', required=True)    
    parser.add_argument('-o', '--output_path', type=str, help='Output image path', required=True)
    parser.add_argument('-d', '--detect_faces', type=str, default='True', help='Detect faces True/False (default: True)')
    args = parser.parse_args()

    image_path = args.input_path
    output_path = args.output_path    
    need_detect_faces = args.detect_faces

    image = cv2.imread(image_path)

    if(need_detect_faces == 'False'):
        gender, confidence = predict_gender(image)
        print(f'{gender} with confidence {confidence}')
        exit(0)
    
    predictions = analyze_image(image)
    new_image = draw_res(image, predictions)
    cv2.imwrite(output_path, new_image)
    print('Result image saved in: ' + output_path)

if __name__ == "__main__":
    __main__(sys.argv)