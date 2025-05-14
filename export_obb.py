import argparse
import os
import shutil
import fiftyone as fo
from tqdm import tqdm
from ultralytics.utils.ops import xywhr2xyxyxyxy, xyxyxyxy2xywhr
import numpy as np

def get_samples(dataset_name, tag):
    dataset = fo.load_dataset(dataset_name)
    samples = dataset.match_tags(tag)
    return samples

def extend_line(x1y1, x2y2):
    x1, y1 = x1y1
    x2, y2 = x2y2

    # Calculate the slope and y-intercept of the line
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    
    # Calculate y-coordinates when x = 0 and x = 1
    y_at_x0 = intercept
    y_at_x1 = slope + intercept

    return [0.0, y_at_x0] , [1.0, y_at_x1]



def is_close(value, target, margin):
    return abs(value - target) <= margin


def export_samples(samples, split, field, local_dataset_dir):
    os.makedirs(os.path.join(local_dataset_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(local_dataset_dir, "labels", split), exist_ok=True)

    images_dir = os.listdir(os.path.join(local_dataset_dir, "images", split))
    labels_dir = os.listdir(os.path.join(local_dataset_dir, "labels", split))

    min_angle = float("inf")
    max_angle = float("-inf")

    for sample in tqdm(samples):
        # copy and paste filepath
        filename = os.path.basename(sample.filepath)
        if os.path.join(local_dataset_dir, "images", split, filename) in images_dir and os.path.join(local_dataset_dir, "labels", split, filename.split(".")[0] + ".txt") in labels_dir:
            continue
        shutil.copy(sample.filepath, os.path.join(local_dataset_dir, "images", split, filename))

        label_content = []

        height = int(sample.metadata.height * 0.1)
        width = sample.metadata.width

        if sample[field] is not None:

            # Find longest polyline using a lambda function to calculate length
            longest_polyline = max(sample[field].polylines, 
                                key=lambda p: sum(((x2-x1)**2 + (y2-y1)**2)**0.5 
                                                for (x1,y1), (x2,y2) in p.points),
                                default=None)

            if longest_polyline:
                point_pair = longest_polyline.points[0]
                x1y1 = point_pair[0]
                x2y2 = point_pair[1]
                x1y1, x2y2 = extend_line(x1y1, x2y2)

                # ortogonal direction
                xyxyxyxyxy = np.array([x1y1[0], x1y1[1], x2y2[0], x2y2[1], x2y2[0], x2y2[1], x1y1[0], x1y1[1]]).reshape(2,4) * width
                xyxyxyxyxy = np.expand_dims(xyxyxyxyxy, axis=0).astype(np.int32) # xyxyxyxyxy2xywhr needs shape [..., 4, 2] and int32
                xywhr = xyxyxyxy2xywhr(xyxyxyxyxy)
                xywhr[:, 3] = int(width * 0.1) # add height so that iou can be calculated
                    
                xyxyxyxyxy = xywhr2xyxyxyxy(xywhr).astype(np.float32)/width
                xyxyxyxyxy = np.clip(xyxyxyxyxy, 0, 1)
                xyxyxyxyxy = np.squeeze(xyxyxyxyxy, axis=0).reshape(-1)
                x1, y1, x2, y2, x3, y3, x4, y4 = xyxyxyxyxy
                obb_cls = [0, x1, y1, x2, y2, x3, y3, x4, y4]
                label_content.append(obb_cls)

                with open(os.path.join(local_dataset_dir, "labels", split, filename.split(".")[0] + ".txt"), "w") as f:
                    for line in label_content:
                        f.write(" ".join([str(i) for i in line]) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fo-dataset-name", default="TRAIN_BB_THERMAL_2024_09", help="dataset name")
    parser.add_argument("--train-tag", default="TRAIN_HORIZON_OBB", help="train tag")
    parser.add_argument("--val-tag", default="VAL_HORIZON_OBB", help="val tag")
    parser.add_argument("--field", default="ground_truth_pl", help="polyline field name")
    parser.add_argument("--local-dataset-dir", default="../datasets/horizon-obb-large-ir-uniquness", help="saved dataset dir")
    args = parser.parse_args()
    
    os.makedirs(args.local_dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(args.local_dataset_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.local_dataset_dir, "labels"), exist_ok=True)

    print("🎯 Exporting training subset")
    train_samples = get_samples(args.fo_dataset_name, args.train_tag)
    export_samples(train_samples, "train", args.field, args.local_dataset_dir)

    print("📊 Exporting validation subset")
    val_samples = get_samples(args.fo_dataset_name, args.val_tag)
    export_samples(val_samples, "val", args.field, args.local_dataset_dir)

    print("✅ Done")
