
from PIL import Image
from torchvision import transforms
import numpy
from yolov4_helper import Helper as YoloHelper
from utils.utils import plot_boxes_cv2, myround
from attack_yolo import create_grid_mask, create_astroid_mask, specific_attack 
import cv2
from utility.utils import *
from utility.torch_utils import *
from utility.darknet2pytorch import Darknet
import torch

import configs

use_cuda = False


def get_yolo_boxes(img, m):
    # img = Image.open(image_path).convert('RGB')
    # resize_small = transforms.Compose([
    #     transforms.Resize((configs.yolo_cfg_height, configs.yolo_cfg_width)),
    # ])
    # img1 = resize_small(img)
    # h, w = numpy.array(img).shape[:2]

    # boxes = do_detect(yolo_model, img1, 0.5, 0.4, True)
    # h, w = numpy.array(img).shape[:2]
    # yolo_boxes = [[(box[0] - box[2] / 2.0) * w, (box[1] - box[3] / 2.0) * h,
    #     (box[0] + box[2] / 2.0) * w, (box[1] + box[3] / 2.0) * h] for box in boxes]

    # boxes = sorted(boxes, key=lambda x:(x[2]-x[0])*(x[3]-x[1])) # sort by area
    # return yolo_boxes
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect2(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        # if i == 1:
        #     print('%s: Predicted in %f seconds.' % (image_path, (finish - start)))
    return plot_boxes_cv2(img, boxes[0], class_names=None)

def draw_boxes_with_label(cv2_image, yolo_model):
    resized_image = cv2.resize(cv2_image, (configs.yolo_cfg_width, configs.yolo_cfg_height))

    boxes = do_detect2(yolo_model, resized_image, 0.5, 0.4, True)

    return plot_boxes_cv2(cv2_image, boxes, None, ["prohibitory", "danger", "mandatory", "others"])

def draw_grid_patches(cv2_image, yolo_helper):
    mask, maskSum = create_grid_mask(yolo_helper.darknet_model, cv2_image, 3, 1.0, (configs.data_height, configs.data_width))
    if maskSum != 0:
        success_attack, attack_img = specific_attack([yolo_helper], cv2_image, mask)
        return attack_img
    return cv2_image

# def draw_astroid_patches(cv2_image, yolo_helper):
#     mask = create_astroid_mask(yolo_helper.darknet_model, cv2_image, 1.0, (configs.data_height, configs.data_width))
    
#     success_attack, attack_img = specific_attack([yolo_helper], cv2_image, mask)

#     return attack_img

def draw_astroid_patches(cv2_image, m):
    mask = create_astroid_mask(m, cv2_image, 1.0, (configs.data_height, configs.data_width))
    yolo_helper = YoloHelper()
    success_attack, attack_img = specific_attack([yolo_helper], cv2_image, mask)

    return attack_img


if __name__ == "__main__":
    
    configs.torch_device = "cuda"

    configs.yolo_class_num = 4

    path="./sample_6.mp4"

    m = Darknet('./models/gtsdb.cfg')

    m.print_network()
    m.load_weights('./models/gtsdb_4000.weights')
    # print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    
    #class_names = load_class_names('./x.names')

        
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
    yolo_helper.load_darknet_model(m)
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






