
from PIL import Image
from torchvision import transforms
import numpy
from yolov4_helper import Helper as YoloHelper
import torch
from utils.utils import do_detect, plot_boxes_cv2, myround
from attack_yolo import create_grid_mask, create_astroid_mask, specific_attack 
import cv2

import configs

def get_yolo_boxes(image_path, yolo_model):
    img = Image.open(image_path).convert('RGB')
    resize_small = transforms.Compose([
        transforms.Resize((configs.yolo_cfg_height, configs.yolo_cfg_width)),
    ])
    img1 = resize_small(img)
    h, w = numpy.array(img).shape[:2]

    boxes = do_detect(yolo_model, img1, 0.5, 0.4, True)
    h, w = numpy.array(img).shape[:2]
    yolo_boxes = [[(box[0] - box[2] / 2.0) * w, (box[1] - box[3] / 2.0) * h,
        (box[0] + box[2] / 2.0) * w, (box[1] + box[3] / 2.0) * h] for box in boxes]

    boxes = sorted(boxes, key=lambda x:(x[2]-x[0])*(x[3]-x[1])) # sort by area
    return yolo_boxes

def draw_boxes_with_label(cv2_image, yolo_model):
    resized_image = cv2.resize(cv2_image, (configs.yolo_cfg_width, configs.yolo_cfg_height))

    boxes = do_detect(yolo_model, resized_image, 0.5, 0.4, True)

    return plot_boxes_cv2(cv2_image, boxes, None, ["prohibitory", "danger", "mandatory", "others"])

def draw_grid_patches(cv2_image, yolo_helper):
    mask, maskSum = create_grid_mask(yolo_helper.darknet_model, cv2_image, 3, 1.0, (configs.data_height, configs.data_width))
    if maskSum != 0:
        success_attack, attack_img = specific_attack([yolo_helper], cv2_image, mask)
        return attack_img
    return cv2_image

def draw_astroid_patches(cv2_image, yolo_helper):
    mask = create_astroid_mask(yolo_helper.darknet_model, cv2_image, 1.0, (configs.data_height, configs.data_width))
    
    success_attack, attack_img = specific_attack([yolo_helper], cv2_image, mask)

    return attack_img


if __name__ == "__main__":
    
    configs.torch_device = "cuda"

    configs.yolo_class_num = 4

    path="./videos/germany-1.mp4"
    
    videoCap = cv2.VideoCapture(path) 

    width  = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

    configs.data_height = height
    configs.data_width = width

    configs.yolo_cfg_width = myround(width, 32)
    configs.yolo_cfg_height = myround(height, 32)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter("test.mp4", fourcc, 30, (width, height))

    success, frame = videoCap.read()
    yolo_helper = YoloHelper()
    count = 0
    while success:
        writer.write(draw_grid_patches(frame, yolo_helper).astype(numpy.uint8))
        success, frame = videoCap.read()
        count += 1
        print(count)


    writer.release()
    # boxes = get_boxes(path)
    # print(len(boxes))
    # print(boxes)
    # draw_boxes(path, boxes)






