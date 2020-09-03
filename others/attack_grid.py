import math
from mmdet.apis import init_detector, inference_detector
import torch
from torch import nn
import random
from torch import optim
import numpy
from PIL import Image
from torchvision import transforms
import os
from yolov4_helper import Helper as YoLov4Helper
from faster_helper import Helper as FasterHelper
import cv2
from utils.utils import *
from constant import *

save_image_dir = "../images_p4x4_faster"

def create_center_mask(shape=(500, 500), size=70):
    mask = torch.zeros(*shape, 3)
    mask[250-size//2:250+size//2, 250-size//2:250+size//2, :]=1

    cv2.imwrite("mask.png", mask.numpy()*255)
    return mask

def create_grid_mask(shape=(500, 500), size=50):
    mask = torch.zeros(*shape, 3)
    lines = 4
    for i in range(1, lines+1):
        mask[50: 450, 100*i, :] = 1
        mask[100*i, 50:450, :] = 1
    return mask

def differential_evolutions(populations, i):
    father = populations[i][0]
    uncle1 = populations[int(random.uniform(0, len(populations)-1))][0]
    uncle2 = populations[int(random.uniform(0, len(populations)-1))][0]
    child = []
    for idx in range(len(father)):
        x = father[idx][0] + int(0.1*(uncle1[idx][0]-uncle2[idx][0]))
        y = father[idx][1] + int(0.1*(uncle1[idx][1]-uncle2[idx][1]))
        x = numpy.clip(x, 50, 450)
        y = numpy.clip(y, 50, 450)

        child += [[x,y]]
    return child

def get_patch_img(img, points, ws, patch_size):
    patch_img = torch.zeros(img.shape).float()
    mask = torch.zeros(img.shape).float()

    for i, (x,y) in enumerate(points):
        mask[y-patch_size//2:y+patch_size//2, x-patch_size//2:x+patch_size//2, :] = 1
        patch_img[y-patch_size//2:y+patch_size//2, x-patch_size//2:x+patch_size//2, :] += get_delta(ws[i])
    return img*(1-mask) + patch_img

def create_yolo_object_mask(darknet_model, image_path, shape=(500, 500), size=50):
    mask = torch.zeros(*shape, 3)
    img = Image.open(img_path).convert('RGB')
    resize_small = transforms.Compose([
            transforms.Resize((608, 608)),
    ])
    img1 = resize_small(img)
    h, w = numpy.array(img).shape[:2]

    boxes = do_detect(darknet_model, img1, 0.5, 0.4, True)

    grids = []
    for i, box in enumerate(boxes):
        x1 = (box[0] - box[2] / 2.0) * w
        y1 = (box[1] - box[3] / 2.0) * h
        x2 = (box[0] + box[2] / 2.0) * w
        y2 = (box[1] + box[3] / 2.0) * h
        print(x1, y1, x2, y2)
        grids += [[int(x1), int(y1), int(x2), int(y2)]]
    for x1, y1, x2, y2 in grids:
        x1 = np.clip(x1, 0, 499)
        x2 = np.clip(x2, 0, 499)

        y1 = np.clip(y1, 0, 499)
        y2 = np.clip(y2, 0, 499)
        print("x1, y1, x2, y2", x1, y1, x2, y2)
        y_middle = (y1+y2)//2
        x_middle = (x1+x2)//2
            
        y_interval = max(32, (y2-y1)//5)
        x_interval = max(32, (x2-x1)//5)
        for i in range(1, 5):
            if mask.sum()>4500*3: break
            if y1+i*y_interval>y2: break
            mask[np.clip(y1+i*y_interval, 0, 499), x1:x2, :]=1
        for i in range(1, 5):
            if mask.sum()>4500*3: break
            if x1+i*x_interval>x2: break
            mask[y1:y2, np.clip(x1+i*x_interval, 0, 499), :]=1

    print("mask sum", mask.sum())
    return mask

def create_faster_object_mask(faster_model, image_path, shape=(500, 500), size=50):
    result = inference_detector(faster_model, image_path)
    boxes = []
    for box in result:
        if len(box)>0:
            boxes += box.tolist()
    boxes = sorted(boxes, key=lambda x:-x[-1])
    grids = [box[:4] for box in boxes if box[-1]>0.2]
    print(grids)

    """
    for i, box in enumerate(boxes):
        x1 = (box[0] - box[2] / 2.0) * w
        y1 = (box[1] - box[3] / 2.0) * h
        x2 = (box[0] + box[2] / 2.0) * w
        y2 = (box[1] + box[3] / 2.0) * h
        print(x1, y1, x2, y2)
        grids += [[int(x1), int(y1), int(x2), int(y2)]]
    """

    mask = torch.zeros(*shape, 3)
    visited_mask = torch.zeros(*shape, 3)

    for x1, y1, x2, y2 in grids:
        x1 = int(np.clip(x1, 0, 499))
        x2 = int(np.clip(x2, 0, 499))

        y1 = int(np.clip(y1, 0, 499))
        y2 = int(np.clip(y2, 0, 499))

        print("x1, y1, x2, y2", x1, y1, x2, y2)
        y_middle = (y1+y2)//2
        x_middle = (x1+x2)//2

        lines = 4
            
        y_interval = max(32, (y2-y1)//(lines+1))
        x_interval = max(32, (x2-x1)//(lines+1))
        for i in range(1, lines+1):
            if mask.sum()>4500*3: break
            if y1+i*y_interval>y2: break
            tmp_mask = torch.zeros(*shape, 3)
            tmp_mask[np.clip(y1+i*y_interval, 0, 499), x1:x2, :]=1
            before_area = tmp_mask.sum()
            after_area = (tmp_mask*(1-visited_mask)).sum()
            if float(after_area) / float(before_area) < 0.3: continue
            mask = mask + tmp_mask
        for i in range(1, lines+1):
            if mask.sum()>4500*3: break
            if x1+i*x_interval>x2: break
            tmp_mask = torch.zeros(*shape, 3)
            tmp_mask[y1:y2, np.clip(x1+i*x_interval, 0, 499), :]=1
            before_area = tmp_mask.sum()
            after_area = (tmp_mask*(1-visited_mask)).sum()
            if float(after_area) / float(before_area) < 0.3: continue
            mask = mask + tmp_mask

        visited_mask[y1:y2, x1:x2, :] = 1

    print("mask sum", mask.sum())
    return mask

def create_strip_mask(shape=(500,500)):
    mask = torch.zeros(*shape, 3)
    rows = [15*i+125 for i in range(1, 19)]
    for row in rows:
        mask[row, 125:375, :]=1
    mask[125:375, 200, :] = 1
    mask[125:375, 300, :] = 1
    return mask

def get_delta(w):
    w = torch.clamp(w, 0, 255)
    #return (1+(1-torch.exp(w))/(1+torch.exp(w))) * 127 # [0, 2*127] = [0, 254]
    return w

def get_patch_positions(grad, grid_size, patch_num):
    # find max grad positions
    grad = grad.permute(2,0,1).unsqueeze(0)
    #ave_pool = nn.AvgPool2d(grid_size, stride=grid_size)
    max_pool = nn.MaxPool2d(grid_size, stride=grid_size)
    #grad = ave_pool(grad)
    grad = torch.abs(grad)
    grad = grad.sum(dim=1).unsqueeze(1)
    grad = max_pool(grad)
    grad_1 = grad
    grad = torch.abs(grad.squeeze().view(-1))
    grad_topk = grad.topk(patch_num)
    positions = []
    for item in grad_topk[1]:
        mod_value = 500 // grid_size
        x = item.item() % mod_value 
        y = item.item() // mod_value

        x *= grid_size
        y *= grid_size
        positions += [[y+grid_size//2, x+grid_size//2]]
    return positions

def create_mask_by_positions(positions, patch_size=20, shape=(500, 500)):
    mask = torch.zeros(*shape, 3)
    for position in positions:
        y, x = position
        mask[y-patch_size//2:y+patch_size//2, x-patch_size//2:x+patch_size//2] = 1
    return mask



def universal_attack(model_helpers, img_paths, mask):
    eps = 1
    epoches = 100
    w = torch.zeros(500, 500, 3).float()+127
    w.requires_grad = True
    for epoch in range(epoches):
        loss_sum, objects_sum = 0, 0

        for t, img_path in enumerate(img_paths):
            img = cv2.imread(img_path)
            img = torch.from_numpy(img).float()
            patch_img = img * (1-mask) + get_delta(w)*mask
            patch_img = patch_img.to(device)

            attack_loss = 0
            object_nums = 0
            for model_helper in model_helpers:
                al, on = model_helper.attack_loss(patch_img)
                attack_loss += al
                object_nums += on
            loss_sum += attack_loss.item()
            objects_sum += object_nums


            if t%20==0: print("t: {}, attack_loss:{}, object_nums:{}".format(t, attack_loss, object_nums))
            attack_loss.backward()
            w = w - eps * w.grad.sign()
            w = w.detach()
            w.requires_grad = True

        cv2.imwrite("patch_{}.png".format(epoch), w.detach().cpu().numpy())
        with open("log.txt", "a") as infile:
            infile.write("loss_sum:{}, objects_sum:{}\n".format(loss_sum, objects_sum))


if __name__ == "__main__":

    random.seed(30)

    #mask = create_center_mask()
#    mask = create_grid_mask()
    yolov4_helper = YoLov4Helper()
    faster_helper = FasterHelper()
    model_helpers = [yolov4_helper, faster_helper]

    mask = create_grid_mask()
    img_paths = [os.path.join("../images", img_path) for img_path in os.listdir("../images")]

    universal_attack(model_helpers, img_paths, mask)
    exit(0)


    #model_helpers = [yolov4_helper]
    #model_helpers = [faster_helper]

    """
    for img_path in os.listdir("../images_p"):
        img_path = os.path.join("../images_p", img_path)
        create_object_mask(yolov4_helper.darknet_model, img_path)

    exit(0)
    img_path = "../images_p/1013.png"
    mask = create_object_mask(yolov4_helper.darknet_model, img_path)
    """
    success_count = 0
    os.system("mkdir -p {}".format(save_image_dir))

    for i, img_path in enumerate(os.listdir("../images")):
        img_path_ps = os.listdir(save_image_dir)
        if img_path in img_path_ps or img_path.replace(".", "_fail.") in img_path_ps: continue
        print("img_path", img_path)
            
        img_path = os.path.join("../images", img_path)
        #mask = create_yolo_object_mask(yolov4_helper.darknet_model, img_path)
        mask = create_faster_object_mask(faster_helper.model, img_path)
        #mask = create_strip_mask()
        #mask = create_center_mask()
        success_attack = specific_attack(model_helpers, img_path, mask)
        if success_attack: success_count += 1
        print("success: {}/{}".format(success_count, i))

            


