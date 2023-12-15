#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--datafolder', '-d', help='Path to input', required=False)
parser.add_argument("--predfolder", "-p", required=True)
parser.add_argument("--area", type=int, default=None)
args = parser.parse_args()
print(args)

try:
    import valeodata
    data_root = valeodata.download("S3DIS_aligned_processed")
    args.datafolder = data_root
    print("Datafolder {}".format(args.datafolder))
except:
    print("No valeo cluster functions avilable")
    args.datafolder = "../../../../data/s3dis/processed_data/"

gt_label_filenames = []
pred_label_filenames = []
PRED_DIR = os.listdir(args.predfolder)

if args.area is not None:
    PRED_DIR = ["Area_{}".format(args.area)]

for area in PRED_DIR:
    Rooms = os.listdir(os.path.join(args.predfolder, area))
    for room in Rooms:
        path_gt_label = os.path.join(args.datafolder, area, room, 'label.npy')
        path_pred_label = os.path.join(args.predfolder, area, room, 'pred.txt')
        pred_label_filenames.append(path_pred_label)
        gt_label_filenames.append(path_gt_label)

num_room = len(gt_label_filenames)


print(num_room)
print(len(pred_label_filenames))
assert num_room == len(pred_label_filenames)

gt_classes = [0 for _ in range(13)]
positive_classes = [0 for _ in range(13)]
true_positive_classes = [0 for _ in range(13)]
confusion_matrix = np.zeros((13, 13))

for i in range(num_room):
    print(i, "/"+str(num_room))
    print(pred_label_filenames[i])
    pred_label = np.loadtxt(pred_label_filenames[i])
    gt_label = np.load(gt_label_filenames[i])
    for j in range(gt_label.shape[0]):
        gt_l = int(gt_label[j])
        pred_l = int(pred_label[j])
        gt_classes[gt_l] += 1
        positive_classes[pred_l] += 1
        true_positive_classes[gt_l] += int(gt_l == pred_l)
        confusion_matrix[gt_l, pred_l] = confusion_matrix[gt_l, pred_l] +1

#Print and save results
result_path = os.path.join(args.predfolder, "results.txt")
print("Result path {}".format(result_path))
with open(result_path, "w+") as file:
    print(gt_classes)
    print(positive_classes)
    print(true_positive_classes)
    file.write("gt_classes: {}\n".format(gt_classes))
    file.write("positive_classes: {}\n".format(positive_classes))
    file.write("true_positive_classes: {}\n".format(true_positive_classes))

    print('Overall accuracy: {0}'.format(sum(true_positive_classes)/float(sum(positive_classes))))
    file.write("Overall_acc:: {}\n".format(sum(true_positive_classes)/float(sum(positive_classes))))
    file.write("IoU:\n")
    print('IoU:')
    iou_list = []
    for i in range(13):
        iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i])
        print(iou)
        file.write(" {}\n".format(iou))
        iou_list.append(iou)

    Seen_classes = [0, 1, 2, 3, 6, 8, 9, 11, 12]
    Unseen_classes = [4, 5, 7, 10]
    Seen_iou = [iou_list[i] for i in Seen_classes]
    Unseen_iou = [iou_list[i] for i in Unseen_classes]
    mIoU_Seen = sum(Seen_iou) / len(Seen_iou)
    mIoU_Unseen = sum(Unseen_iou) / len(Unseen_iou)
    HmIoU = 2 * mIoU_Seen * mIoU_Unseen / (mIoU_Seen + mIoU_Unseen)

    print("Seen mIoU: {}".format(mIoU_Seen))
    file.write("Seen mIoU: {}\n".format(mIoU_Seen))

    print("Unseen mIoU: {}".format(mIoU_Unseen))
    file.write("Unseen mIoU: {}\n".format(mIoU_Unseen))

    print("All mIoU: {}".format(sum(iou_list) / 13.0))
    file.write("All mIoU: {}\n".format(sum(iou_list) / 13.0))

    print("HmIoU: {}".format(HmIoU))
    file.write("HmIoU: {}\n".format(HmIoU))

    print('********************************************************')
    print("Confusion matrix: {}".format(confusion_matrix))
    file.write('********************************************************')
    file.write("Confusoin matrix: {}\n".format(confusion_matrix))
