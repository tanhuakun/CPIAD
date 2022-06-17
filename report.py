import os
from utils.utils import bbox_iou


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

            box.append(int(split[7]))

            if frame_num not in frames_dict:
                frames_dict[frame_num] = []

            frames_dict[frame_num].append(box)

            
        return frames_dict
                
                
    def find_log_diff(self, file_path1, file_path2, iou_thres=0.75):
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
                        if box1[6] != box2[6]:
                            # diff class, store both classes in box 1
                            box1.append(box2[6])
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


if __name__ == "__main__":
    logger = Boxes_Logger()
    
    results = logger.find_log_diff("test.txt", "test2.txt")

    for result in results:
        print(result)