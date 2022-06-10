
from PIL import Image
from torchvision import transforms
import numpy
#from mmdet.apis import init_detector, inference_detector
#from faster_helper import Helper as FasterHelper
from yolov4_helper import Helper as YoloHelper
import torch
from utils.utils import do_detect, plot_boxes_cv2
from utils.utils_coco import get_coco_label_names
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
'''
def get_faster_boxes(image_path, faster_model=FasterHelper().model):
    result = inference_detector(faster_model, image_path)
    boxes = []
    for box in result:
        if len(box)>0:
            boxes += box.tolist()


    boxes = sorted(boxes, key=lambda x:x[-1])
    boxes = [box for box in boxes if box[-1]>0.25]
    #boxes = sorted(boxes, key=lambda x:(x[2]-x[0])*(x[3]-x[1])) # sort by area

    return boxes
'''
def get_boxes(image_path):
    boxes = get_yolo_boxes(image_path, YoloHelper().darknet_model)# + get_faster_boxes(image_path)
    boxes = sorted(boxes, key=lambda x:(x[2]-x[0])*(x[3]-x[1])) # sort by area
#    print(boxes)
    return boxes

def draw_boxes(image_path, boxes):
    image = cv2.imread(image_path)

    for box in boxes:
        image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

    cv2.imwrite("test2.png", image)

def draw_boxes_with_label(image_path, yolo_model):
    img = Image.open(image_path).convert('RGB')
    resize_small = transforms.Compose([
        transforms.Resize((608, 608)),
    ])
    img1 = resize_small(img)
    h, w = numpy.array(img).shape[:2]

    boxes = do_detect(yolo_model, img1, 0.5, 0.4, True)

    cv2_image = cv2.imread(image_path)
    plot_boxes_cv2(cv2_image, boxes, "test2.png", ["prohibitory", "danger", "mandatory", "others"])


if __name__ == "__main__":
    
    configs.torch_device = "cpu"
    path="./images2/00021.jpg"
    
    # boxes = get_boxes(path)
    # print(len(boxes))
    # print(boxes)
    # draw_boxes(path, boxes)
    
    draw_boxes_with_label(path,  YoloHelper().darknet_model)





