import pandas as pd
from pathlib import Path, PurePath
import os
import argparse
from pprint import pprint 

# /data3/awong/kitti/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02
# /data3/awong/kitti_raw/2011_09_26/2011_09_26_drive_0001_sync/image_02/data


def convert_depth_image_path(path):
    p = Path(path).parts
    
    date = p[0][0:10]
    directory = p[0]
    img_dir = p[-2]
    img = p[-1]
    
    return f"{date}/{directory}/{img_dir}/data/{img}"

def convert_all_depth_images(paths):
    img_paths = []
    for path in paths:
        img_paths.append(convert_depth_image_path(path))
    return img_paths

def get_files(img_dir):
    img_paths = []
    for path, _, files in os.walk(img_dir):
        for name in files:
            if ".png" in name:
                img_paths.append(PurePath(path, name).__str__().replace(img_dir, ""))
    return img_paths
            
def create_csv(filename, image_path, depth_path: str):
    depth_images = get_files(depth_path)
    rgb_images = convert_all_depth_images(depth_images)
    verify_img_exists(image_path, depth_images)
    
    df = pd.DataFrame.from_dict({'images': rgb_images, 'depth': depth_images})
    df.to_csv(filename)

def verify_img_exists(img_path, depth_images):
    for d_path in depth_images:
        path = convert_depth_image_path(d_path)
        if not os.path.exists(img_path + path):
            print(f"File not found: {path}")
            exit(1)
        else:
            print(f"Found file: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate kitti dataset .csv files for training/validation/test")
    parser.add_argument('filename', help="filename of output .csv")
    parser.add_argument('depth', help="depth image path to kitti")
    parser.add_argument('raw', help='raw image path')
    
    args = parser.parse_args()
    
    create_csv(args.filename, args.raw, args.depth)
    
    # depth_images = get_files(args.depth)
    # # pprint(depth_images)
    # rgb_images = convert_all_depth_images(depth_images)
    # # pprint(rgb_images)
    # verify_img_exists(args.raw, depth_images)
    # # create_csv()