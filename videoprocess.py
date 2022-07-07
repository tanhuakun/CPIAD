
from PIL import Image
from torchvision import transforms
import numpy
from yolov4_helper import Helper as YoloHelper
import torch
from utils.utils import do_detect, load_class_names, plot_boxes_cv2, myround
from attack_yolo import create_grid_mask, create_astroid_mask, specific_attack 
import cv2

import configs
import argparse

def get_yolo_boxes(image_path, yolo_model):
    img = Image.open(image_path).convert('RGB')
    resize_small = transforms.Compose([
        transforms.Resize((configs.yolo_resize_height, configs.yolo_resize_width)),
    ])
    img1 = resize_small(img)
    h, w = numpy.array(img).shape[:2]

    boxes = do_detect(yolo_model, img1, 0.5, 0.4, True)
    h, w = numpy.array(img).shape[:2]
    yolo_boxes = [[(box[0] - box[2] / 2.0) * w, (box[1] - box[3] / 2.0) * h,
        (box[0] + box[2] / 2.0) * w, (box[1] + box[3] / 2.0) * h] for box in boxes]

    boxes = sorted(boxes, key=lambda x:(x[2]-x[0])*(x[3]-x[1])) # sort by area
    return yolo_boxes

def draw_boxes_with_label(cv2_image, yolo_model, classes_list):
    recoloured = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    boxes = do_detect(yolo_model, recoloured, 0.45, 0.4, True)
    return plot_boxes_cv2(cv2_image, boxes, None, classes_list)

def draw_grid_patches(cv2_image, yolo_helper):
    mask, maskSum = create_grid_mask(yolo_helper.darknet_model, cv2_image, 3, 1.0, (configs.data_height, configs.data_width))
    if maskSum != 0:
        success_attack, attack_img = specific_attack([yolo_helper], cv2_image, mask)
        return attack_img
    return cv2_image

# Not effective!!!
# def draw_astroid_patches(cv2_image, yolo_helper):
#     mask = create_astroid_mask(yolo_helper.darknet_model, cv2_image, 1.0, (configs.data_height, configs.data_width))
    
#     success_attack, attack_img = specific_attack([yolo_helper], cv2_image, mask)

#     return attack_img
TYPE_ATTACK = "attack"
TYPE_DETECT = "detect"
SUPPORTED_ACTIONS = [TYPE_ATTACK, TYPE_DETECT]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str)

    parser.add_argument('--video_src', type=str, default="./videos/sample_6.mp4")
    parser.add_argument('--weights', type=str, default="./models/gtsdb_4000.weights")
    parser.add_argument('--cfg', type=str, default="./models/gtsdb.cfg")
    parser.add_argument('--outfile', type=str, default="video_out.mp4")
    parser.add_argument("--classes", type=str, default="./models/classes.txt")
    args = parser.parse_args()
    
    if args.action not in SUPPORTED_ACTIONS:
        raise Exception("Please select an action!")


    configs.torch_device = "cpu"
    
    videoCap = cv2.VideoCapture(args.video_src)

    class_names = load_class_names(args.classes)
    configs.yolo_class_num = len(class_names)

    width  = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    fps = float(videoCap.get(cv2.CAP_PROP_FPS))
    # storing original video height
    configs.data_height = height
    configs.data_width = width
    # round off both width and height to nearest 32 to pass into yolo algorithm.
    configs.yolo_resize_width = myround(width, 32)
    configs.yolo_resize_height = myround(height, 32)
    # simple and easy mp4v, welcome to try others.
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(args.outfile, fourcc, fps, (width, height))
    yolo_helper = YoloHelper(args.cfg, args.weights)



    if args.action == TYPE_ATTACK:
        video_gen = lambda frame : draw_grid_patches(frame, yolo_helper)
    elif args.action == TYPE_DETECT:
        
        video_gen = lambda frame: draw_boxes_with_label(frame, yolo_helper.darknet_model, class_names)
    
    success, frame = videoCap.read()
    count = 0
    while success:
        writer.write(video_gen(frame).round().astype(numpy.uint8))
        success, frame = videoCap.read()
        count += 1
        print(count)

    writer.release()







