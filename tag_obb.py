"""
Tag a FiftyOne dataset with a balanced mix of annotated and noise samples,
then split the selected samples into training and validation sets using a specified grouping.

This script filters the dataset based on camera names, selects a fixed number of samples
with a desired ratio of noise (no horizon annotations), and assigns tags to the selected
samples as `TRAIN_<tag_name>` and `VAL_<tag_name>` based and by grouping by "trip" or "sequence".

Usage (example):

    python tag_obb.py \
        --dataset-name TRAIN_RGB_ALL_2025_02_IMAGE_BB \
        --tags-suffix HORIZON_OBB_XLARGE \
        --total-images 50000 \
        --val-ratio 0.1 \
        --noise-ratio 0.2 \
        --camera-list Offshore_e-CAM50_CUNX_1280x960_1 ELP_1080_18mm_1 \
        --split-by sequence \
        --horizon-field ground_truth_pl

Arguments:
    --dataset-name:   Name of the FiftyOne dataset
    --tags-suffix:    Suffix used in tags (e.g. "MYTASK" → tags: TRAIN_MYTASK, VAL_MYTASK)
    --total-images:   Total number of samples to select
    --val-ratio:      Proportion of validation data (e.g. 0.25 = 25%)
    --noise-ratio:    Proportion of noise (unannotated) samples (e.g. 0.2 = 20%)
    --camera-list:    List of camera names to include (e.g. thermal_wide thermal_narrow)
    --split-by:       Field to group on when splitting (e.g. sequence or trip)
    --horizon-field:  Field containing horizon annotations (default: ground_truth_pl)

Note:
- If the tags TRAIN_<tag_name> or VAL_<tag_name> already exist in the dataset, the script will abort.
- You can manually remove those tags using `clean_existing_tags(dataset, tag_name)` in Python.

"""


import argparse

import fiftyone as fo
from fiftyone import ViewField as F


import argparse
import os

os.environ["FIFTYONE_CONFIG_PATH"] = str("/etc/fiftyone/config.json")

import shutil
from pathlib import Path
import fiftyone.brain as fob
import yaml
import random

def get_annotated_trip_list(dataset):
    """
    Trips only started to be annotated after trip 140
    """
    annotated_trips = []
    all_trips = dataset.distinct("trip")
    print(len(all_trips))
    for trip in all_trips:
        if "Trip" in trip:
            print(trip)
            if int(trip.split("_")[-1]) > 140:
                annotated_trips.append(trip)
        else:
            annotated_trips.append(trip)
    print(len(annotated_trips))
    return annotated_trips

def get_filtered_view(dataset, camera_list):
    """
    Filters the dataset by the specified camera names.
    """
    return dataset.match(F("camera_name").is_in(camera_list))


def subsample_view(view: fo.DatasetView, horizon_field: str, total_count: int, noise_ratio: float):
    """
    Subsample the dataset view to contain a balanced number of annotated and noise samples.
    """
    annotated_trip_list = get_annotated_trip_list(view)
    view = view.match(F("trip").is_in(annotated_trip_list))

    annotated = view.exists(f"{horizon_field}", True)
    noise = view.exists(f"{horizon_field}", False)
    
    print(f"🎲 Found {len(annotated)} annotated samples and {len(noise)} noise samples out of {len(view)} samples")

    n_noise = int(total_count * noise_ratio)
    n_annotated = total_count - n_noise

    if len(noise) < n_noise:
        raise ValueError(f"Not enough noise samples: needed {n_noise}, found {len(noise)}")
    if len(annotated) < n_annotated:
        raise ValueError(f"Not enough annotated samples: needed {n_annotated}, found {len(annotated)}")

    sampled_noise = noise.sort_by("uniqueness", reverse=True).limit(n_noise)
    sampled_annotated = annotated.sort_by("uniqueness", reverse=True).limit(n_annotated)

    return sampled_annotated.concat(sampled_noise)



def split_by_field(
    dataset: fo.DatasetView, field: str, val_ratio: float, tags_suffix: str
) -> fo.DatasetView:
    """
    Split and tag the dataset into train and validation sets based on a field.
    Tags used: TRAIN_{tags_suffix}, VAL_{tags_suffix}
    """
    counts = dataset.count_values(field)
    counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    val_keys = list(counts.keys())[:: int(1 / val_ratio)]

    train_n = sum(v for k, v in counts.items() if k not in val_keys)
    val_n = sum(v for k, v in counts.items() if k in val_keys)
    total_n = train_n + val_n
    print(f"📊 Ratio train/val split: {train_n/total_n:.2f}:{val_n/total_n:.2f}")

    # Apply new tags
    dataset.match(~F(field).is_in(val_keys)).tag_samples(f"TRAIN_{tags_suffix}")
    dataset.match(F(field).is_in(val_keys)).tag_samples(f"VAL_{tags_suffix}")

    print(f"✅ Tagged {train_n} samples as TRAIN_{tags_suffix}, {val_n} as VAL_{tags_suffix}")
    return dataset


def clean_existing_tags(dataset: fo.Dataset, tag_suffix: str):
    """
    Removes existing TRAIN_{tag_suffix} and VAL_{tag_suffix} tags from the dataset.
    """
    train_tag = f"TRAIN_{tag_suffix}"
    val_tag = f"VAL_{tag_suffix}"

    dataset.untag_samples(train_tag)
    dataset.untag_samples(val_tag)
    print(f"🧹 Removed existing tags: {train_tag}, {val_tag}")



def tag_horizon_dataset(
    dataset_name,
    tags_suffix,
    total_images,
    val_ratio,
    noise_ratio,
    camera_list,
    split_by,
    horizon_field,
):
    dataset = fo.load_dataset(dataset_name)

    # Check for existing tags
    existing_tags = dataset.distinct("tags")
    print(f"📊 Existing tags: {existing_tags}")
    if f"TRAIN_{tags_suffix}" in existing_tags or f"VAL_{tags_suffix}" in existing_tags:
        # print(f"❌ Aborting: Tag 'TRAIN_{tags_suffix}' or 'VAL_{tags_suffix}' already exists in dataset '{dataset_name}'")
        # print(f"❌ Please call clean_existing_tags() or use a different suffix name.")
        # return
        print(f"🧹 Removing existing tags: TRAIN_{tags_suffix}, VAL_{tags_suffix}")
        clean_existing_tags(dataset, f"TRAIN_{tags_suffix}")
        clean_existing_tags(dataset, f"VAL_{tags_suffix}")
        print(f"🧹 Removed existing tags: TRAIN_{tags_suffix}, VAL_{tags_suffix}")

    print("🎯 Filtering by camera...")
    view = get_filtered_view(dataset, camera_list)

    print(f"🎲 Subsampling {total_images} samples with {int(total_images * noise_ratio)} noise...")
    samples = subsample_view(view, horizon_field, total_images, noise_ratio)

    print(f"📦 Splitting by '{split_by}' with val ratio = {val_ratio}")
    split_by_field(samples, split_by, val_ratio, tags_suffix)


def parse_args():
    parser = argparse.ArgumentParser(description="Tag a FiftyOne dataset with balanced noise and train/val splits")
    parser.add_argument("--dataset-name", type=str, required=True, help="Name of the FiftyOne dataset")
    parser.add_argument("--tags-suffix", type=str, default="OBB_HORIZON", required=True, help="Suffix used in TRAIN_/VAL_ tags (e.g. MYTAG → TRAIN_MYTAG)")
    parser.add_argument("--total-images", type=int, default=10000, help="Total number of samples to tag")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (0.2 = 20%%)")
    parser.add_argument("--noise-ratio", type=float, default=0.0, help="Noise (no-horizon) ratio (0.2 = 20%%)")
    parser.add_argument("--camera-list", type=str, nargs="+", required=True, help="List of camera names to include")
    parser.add_argument("--split-by", type=str, default="sequence", help="DB field to split the training data")
    parser.add_argument("--horizon-field", type=str, default="ground_truth_pl", help="Label field to check for horizon detections")
    return parser.parse_args()


def main():
    args = parse_args()
    tag_horizon_dataset(
        dataset_name=args.dataset_name,
        tags_suffix=args.tags_suffix,
        total_images=args.total_images,
        val_ratio=args.val_ratio,
        noise_ratio=args.noise_ratio,
        camera_list=args.camera_list,
        split_by=args.split_by,
        horizon_field=args.horizon_field,
    )


if __name__ == "__main__":
    main()