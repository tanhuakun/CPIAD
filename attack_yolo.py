import math
import asyncio
from cv2 import COLOR_RGB2BGR
#from mmdet.apis import init_detector, inference_detector
import torch
from torch import nn
import random
from torch import optim
import numpy
from PIL import Image
from torchvision import transforms
import os
from yolov4_helper import Helper as YoLov4Helper
#from faster_helper import Helper as FasterHelper
import cv2
from utils.utils import *
from constant import *
import argparse

import configs


def create_astroid_mask(darknet_model, img, box_scale, shape=(500, 500)):
    mask = torch.zeros(*shape, 3)
    
    h, w = img.shape[:2]

    img1 = cv2.resize(img, (configs.yolo_resize_width, configs.yolo_resize_height))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    boxes = do_detect(darknet_model, img1, 0.35, 0.4, False)

    yolo_boxes = [[(box[0] - box[2] / 2.0) * w, (box[1] - box[3] / 2.0) * h,
        (box[0] + box[2] / 2.0) * w, (box[1] + box[3] / 2.0) * h] for box in boxes]
    boxes = yolo_boxes
    boxes = sorted(boxes, key=lambda x:(x[2]-x[0])*(x[3]-x[1])) # sort by area
    grids = boxes
    

    '''
    result = inference_detector(faster_model, image_path)
    boxes = []
    for box in result:
        if len(box)>0:
            boxes += box.tolist()
    
    boxes = [box[:4] for box in boxes if box[-1]>0.3]
    #boxes += yolo_boxes
    boxes = sorted(boxes, key=lambda x:(x[2]-x[0])*(x[3]-x[1])) # sort by area
    grids = boxes
    '''


    mask = torch.zeros(*shape, 3)
    visited_mask = torch.zeros(*shape, 3)
    num = 0
    for _, (x1, y1, x2, y2) in enumerate(grids):
        if num>9: break
        x1 = int(np.clip(x1, 0, configs.data_width))
        x2 = int(np.clip(x2, 0, configs.data_width))

        y1 = int(np.clip(y1, 0, configs.data_height))
        y2 = int(np.clip(y2, 0, configs.data_height))

        print("x1, y1, x2, y2", x1, y1, x2, y2)
        y_middle = (y1+y2)//2
        x_middle = (x1+x2)//2

        # shrink box
        box_h, box_w = int((y2-y1)*box_scale), int((x2-x1)*box_scale)
        y11 = y_middle-box_h//2
        y22 = y_middle+box_h//2
        x11 = x_middle-box_w//2
        x22 = x_middle+box_w//2


        cross_line_x_len = x_middle-x11
        cross_line_y_len = y_middle-y11
        cross_line_len = max(y_middle-y11, x_middle-x11)
        y_step, x_step = cross_line_y_len/cross_line_len, cross_line_x_len/cross_line_len

        tmp_mask = torch.zeros(mask.shape)
        tmp_mask[y_middle, x11:x22, :] = 1
        tmp_mask[y11:y22, x_middle, :] = 1
        for i in range(1, cross_line_len):
            tmp_mask[y_middle-int(i*y_step), x_middle-int(i*x_step), :] = 1
            tmp_mask[y_middle+int(i*y_step), x_middle-int(i*x_step), :] = 1
            tmp_mask[y_middle-int(i*y_step), x_middle+int(i*x_step), :] = 1
            tmp_mask[y_middle+int(i*y_step), x_middle+int(i*x_step), :] = 1
        before_area = tmp_mask.sum()
        after_area = (tmp_mask*(1-visited_mask)).sum()
        if float(after_area) / float(before_area) < 0.5:
            continue

        if (mask+tmp_mask).sum()>5000*3: break
        num += 1
        mask = mask + tmp_mask
        visited_mask[y1:y2, x1:x2, :] = 1
    print("mask sum", mask.sum())
    return mask


def create_grid_mask(darknet_model, img, lines=3, box_scale=1.0, shape=(500, 500)):
    mask = torch.zeros(*shape, 3)
    # get yolo bounding boxes <-- this was commented out originally
    h, w = img.shape[:2]

    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = do_detect(darknet_model, img1, 0.35, 0.4, True)

    yolo_boxes = [[(box[0] - box[2] / 2.0) * w, (box[1] - box[3] / 2.0) * h, 
        (box[0] + box[2] / 2.0) * w, (box[1] + box[3] / 2.0) * h] for box in boxes]
    boxes = yolo_boxes
    grids = boxes

    # get boxes from the rcnn
    '''
    result = inference_detector(faster_model, image_path)
    boxes = []
    for box in result:
        if len(box)>0:
            boxes += box.tolist()

    boxes = [box[:4] for box in boxes if box[-1]>0.3]
    #boxes += yolo_boxes
    boxes = sorted(boxes, key=lambda x:(x[2]-x[0])*(x[3]-x[1])) # sort by area
    grids = boxes
    '''
    mask = torch.zeros(*shape, 3)
    visited_mask = torch.zeros(*shape, 3)

    for x1, y1, x2, y2 in grids:
        x1 = int(np.clip(x1, 0, configs.data_width))
        x2 = int(np.clip(x2, 0, configs.data_width))

        y1 = int(np.clip(y1, 0, configs.data_height))
        y2 = int(np.clip(y2, 0, configs.data_height))

        print("x1, y1, x2, y2", x1, y1, x2, y2)
        y_middle = (y1+y2)//2
        x_middle = (x1+x2)//2
        # shrink box
        box_h, box_w = int((y2-y1)*box_scale), int((x2-x1)*box_scale)
        y11 = y_middle-box_h//2
        y22 = y_middle+box_h//2
        x11 = x_middle-box_w//2
        x22 = x_middle+box_w//2

        #min_interval = 32
        #if lines == 0:
        #    min_interval = 0

#        y_interval = max(min_interval, (y2-y1)//(lines+1))
#        x_interval = max(min_interval, (x2-x1)//(lines+1))
 

        ### is min_interval important???
        y_interval = (y2-y1)//(lines+1)
        x_interval = (x2-x1)//(lines+1)        
        for i in range(1, lines+1):
            if mask.sum()>4500*3: break
            if y1+i*y_interval>y2: break
            tmp_mask = torch.zeros(*shape, 3)
            tmp_mask[np.clip(y1+i*y_interval, 0, configs.data_height), x11:x22, :]=1
            before_area = tmp_mask.sum()    
            after_area = (tmp_mask*(1-visited_mask)).sum()
            if float(after_area) / float(before_area) < 0.5: continue 
            mask = mask + tmp_mask
        for i in range(1, lines+1):
            if mask.sum()>4500*3: break
            if x1+i*x_interval>x2: break
            tmp_mask = torch.zeros(*shape, 3)
            tmp_mask[y11:y22, np.clip(x1+i*x_interval, 0, configs.data_width), :]=1
            before_area = tmp_mask.sum()
            after_area = (tmp_mask*(1-visited_mask)).sum()
            if float(after_area) / float(before_area) < 0.5: continue
            mask = mask + tmp_mask

        visited_mask[y1:y2, x1:x2, :] = 1

    maskSum = mask.sum()
    print("mask sum", maskSum)
    return mask, maskSum

def get_delta(w):
    w = torch.clamp(w, 0, 255)
    #return (1+(1-torch.exp(w))/(1+torch.exp(w))) * 127 # [0, 2*127] = [0, 254]
    return w
'''
async def get_attack_loss(helper, img):
    al, on = await helper.attack_loss(img)
    return al, on
'''
def get_attack_loss(helper, img):
    al, on = helper.attack_loss(img)
    return al, on

def specific_attack(model_helpers, img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float()

    t, max_iterations = 0, 100
    stop_loss = 1e-6
    eps = 1
    w = torch.zeros(img.shape).float()+127
    w.requires_grad = True
    success_attack = False
    patch_size = 70
    patch_num = 10
    min_object_num = 1000
    min_img = img
    grads = 0
    #loop = asyncio.get_event_loop()
    success_count = 0

    min_loss = 10000

    while t<max_iterations or success_attack:
        t+=1

        # check connectivity
        patch_connecticity = torch.abs(get_delta(w)-img).sum(-1)
        patch_connecticity = (patch_connecticity==0)
        patch = get_delta(w)
        patch[patch_connecticity] += 1

        patch_img = img * (1-mask) + patch*mask
        patch_img = patch_img.to(configs.torch_device)

        attack_loss = 0
        object_nums = 0
        # https://xubiubiu.com/2019/06/12/python3-%E8%8E%B7%E5%8F%96%E5%8D%8F%E7%A8%8B%E7%9A%84%E8%BF%94%E5%9B%9E%E5%80%BC/
        """tasks = [
                get_attack_loss(model_helpers[0], patch_img),
                #get_attack_loss(model_helpers[1], patch_img),
                ]
        res = loop.run_until_complete(asyncio.gather(*tasks))"""
        attack_loss, object_nums = get_attack_loss(model_helpers[0], patch_img)
        """for al, on in res:
            attack_loss += al
            object_nums += on"""
        if min_object_num>object_nums:
            min_object_num = object_nums
            min_img = patch_img
            min_loss = attack_loss
        elif min_object_num == object_nums and min_loss > attack_loss:
            min_img = patch_img
            min_loss = attack_loss

        if object_nums==0 and not success_attack:
            success_attack = True
            print("Success attack = True")
        if success_attack:
            success_count += 1
            if success_count >= 8:
                print("LOSS", attack_loss, "NUMS", object_nums)
                break
        if t%5==0: print("t: {}, attack_loss:{}, object_nums:{}".format(t, attack_loss, object_nums))
        attack_loss.backward()
        #grads = grads + w.grad / (torch.abs(w.grad)).sum()
        w = w - eps * w.grad.sign()
        w = w.detach()
        w.requires_grad = True

    min_img = min_img.detach().cpu().numpy()
    
    
    return success_attack, cv2.cvtColor(min_img, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":

    SOURCE_DIR = "images2"
    configs.torch_device = "cpu"
    configs.yolo_resize_width = 960
    configs.yolo_resize_height = 576
    configs.yolo_class_num = 4
    configs.data_width = 1360
    configs.data_height = 800
    random.seed(30)
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_type', type=str, default="grid")
    parser.add_argument('--lines', type=int, default=3)
    parser.add_argument('--box_scale', type=float, default=1.0)
    args = parser.parse_args()
    patch_type = args.patch_type
    lines = args.lines
    box_scale = args.box_scale

    yolov4_helper = YoLov4Helper()
    #faster_helper = FasterHelper()
    #model_helpers = [yolov4_helper, faster_helper]
    model_helpers = [yolov4_helper]
    success_count = 0

    if patch_type == "grid":
        save_image_dir = "images_p_grid_{}x{}_{}".format(lines, lines, box_scale)
    else:
        save_image_dir = "images_p_astroid_{}".format(box_scale)
    os.system("mkdir {}".format(save_image_dir))


    for i, img_name in enumerate(os.listdir(SOURCE_DIR)):
        img_path_ps = os.listdir(save_image_dir)
        '''
        if img_path in img_path_ps:
            success_count+= 1
            continue
        if img_path.replace(".", "_fail.") in img_path_ps: continue
        '''
        print("img_path", img_name)
            
        image_source = os.path.join(SOURCE_DIR, img_name)

        cv2_image = cv2.imread(image_source)

        if patch_type=="grid":
            mask = create_grid_mask(yolov4_helper.darknet_model, cv2_image, lines, box_scale, (configs.data_height, configs.data_width))
        else:
            mask = create_astroid_mask(yolov4_helper.darknet_model, cv2_image, box_scale, (configs.data_height, configs.data_width))
        
        success_attack, attack_img = specific_attack(model_helpers, cv2_image, mask)

        if success_attack:
            path = os.path.join(os.getcwd(), save_image_dir, img_name)
            success_count += 1
        else:
            path = os.path.join(os.getcwd(), save_image_dir, img_name.split(".")[0] + "_fail.png")
        
        if cv2.imwrite(path, attack_img):
            print("Image saved at " + path)
        else:
            print("Image faled to save at " + path)            
                
        print("success: {}/{}".format(success_count, i + 1))
        
            


