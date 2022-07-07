import math
import os

from pyparsing import col
from utils.utils import bbox_iou
import cv2

class Boxes_Logger:

    def open_outfile(self, outfile=None):
        self.outfile = outfile
        self.f = open(outfile, mode="w")

    # normally its 
    def append_yolo_box(self, frame_num, box):
        log = f"{frame_num} {box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]}\n"
        self.f.write(log)

    def close_outfile(self):
        self.f.close()

    def log_to_dict(self, logFile):
        f = open(logFile, 'r')
        frames_dict = {}

        for line in f:
            split = line.strip().split(" ")
            frame_num = int(split[0])
            
            box = []
            for i in range(1,7):
                box.append(float(split[i]))

            #box.append(int(split[7]))

            if frame_num not in frames_dict:
                frames_dict[frame_num] = []

            frames_dict[frame_num].append(box)

            
        return frames_dict
                
                
    def find_log_diff(self, file_path1, file_path2, iou_thres=0.55):
        dict1 = self.log_to_dict(file_path1)
        dict2 = self.log_to_dict(file_path2)

        only_file1 = {}
        only_file2 = {}
        both_files = {}
        both_but_class_diff = {}

        for frame, boxes1 in dict1.items():
            if frame not in dict2:
                only_file1[frame] = boxes1
                continue
            
            boxes2 = dict2[frame]
            #### compare IOUs one by one!
            #### generally we take box1 boundings!
            box2_seen = {}
            for box1 in boxes1:
                box1_matched = False
                for index, box2 in enumerate(boxes2):
                    if index in box2_seen:
                        # we have paired it with a box1 already
                        continue

                    iou = bbox_iou(box1, box2, False)
                    if iou > iou_thres:
                        # likely the same box
                        if box1[5] != box2[5]:
                            # diff class, store both classes in box 1
                            box1.append(box2[5])
                            if frame not in both_but_class_diff:
                                both_but_class_diff[frame] = []
                            both_but_class_diff[frame].append(box1)
                        else:
                            # same class, hence same detection
                            if frame not in both_files:
                                both_files[frame] = []
                            both_files[frame].append(box1)
                        box2_seen[index] = True
                        box1_matched = True
                        break
                    
                    ## not match
                if not box1_matched:
                    if frame not in only_file1:
                        only_file1[frame] = []
                    only_file1[frame].append(box1)

            for index, box2 in enumerate(boxes2):
                if index not in box2_seen:
                    ## boxes not matched to box1....
                    if frame not in only_file2:
                        only_file2[frame] = []
                    only_file2[frame].append(box2)

        return only_file1, only_file2, both_files, both_but_class_diff

def plot_box_color(cv2_image, box, color, text):
    width = cv2_image.shape[1]
    height = cv2_image.shape[0]
    x1 = int((box[0] - box[2] / 2.0) * width)
    y1 = int((box[1] - box[3] / 2.0) * height)
    x2 = int((box[0] + box[2] / 2.0) * width)
    y2 = int((box[1] + box[3] / 2.0) * height)

    cv2_image = cv2.rectangle(cv2_image, (x1, y1), (x2, y2), color, 2)

    t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    text_top = max(y1 - t_size[1], 0)
    cv2.rectangle(cv2_image, (x1,text_top), (min(x1 + t_size[0], width-1), y1), color, -1)
    cv2_image = cv2.putText(cv2_image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
    

    return cv2_image

def plot_boxes_diff(cv2_image, only_file1, only_file2, both_files, both_but_class_diff, classes):
    ## image and list for a particular frame!
    for box in only_file1:
        cv2_image = plot_box_color(cv2_image, box, (26, 255, 26), "blocked")
    for box in only_file2:
        cv2_image = plot_box_color(cv2_image, box, (255, 51, 204), "extra?")
    for box in both_files:
        cv2_image = plot_box_color(cv2_image, box, (0, 51, 255), "still detect")
    for box in both_but_class_diff:
        cv2_image = plot_box_color(cv2_image, box, (0, 102, 255), f"{classes[box[6]]} ->\n{classes[box[7]]}")

    return cv2_image

if __name__ == "__main__":
    logger = Boxes_Logger()
    
    only_file1, only_file2, both_files, both_but_class_diff = logger.find_log_diff( "./videos/benign_log.txt", "./videos/attack_log.txt")
    '''
    path="./videos/high_iter_defense_keep2.mp4"
    
    videoCap = cv2.VideoCapture(path) 

    width  = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter("test diff defense2.mp4", fourcc, 30, (width, height))

    success, frame = videoCap.read()
    count = 0
    while success:
        drawn = plot_boxes_diff(frame,
            only_file1.get(count, []), only_file2.get(count, []), both_files.get(count, []), both_but_class_diff.get(count, []),
            ["prohibitory", "danger", "mandatory", "others"]
        )
        writer.write(drawn)
        success, frame = videoCap.read()
        count += 1
        print(count)

    
    writer.release()
    '''
    item :list 
    only1_total = 0
    for key, item in only_file1.items():
        only1_total += len(item)

    only2_total = 0
    for key, item in only_file2.items():
        only2_total += len(item)

    both_total = 0
    for key, item in both_files.items():
        both_total += len(item)

    class_diff = 0
    for key, item in both_but_class_diff.items():
        class_diff += len(item)

    total_original = only1_total + both_total + class_diff
    percentage =  float(only1_total) / total_original * 100
    print("Total original", total_original)
    print("Total blocked", only1_total)
    print("Percentage blocked", str(percentage) + "%")
    print("Not blocked", both_total + class_diff)
    print("Not blocked and class changed", class_diff)
    print("Extra?", only2_total)