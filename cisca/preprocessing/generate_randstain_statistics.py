import os
import cv2
import numpy as np
import time
import argparse
import yaml
import json
import random
import copy
from tqdm import tqdm
from skimage import color
from fitter import Fitter
import pandas as pd

def getavgstd(image):
    avg = []
    std = []
    image_avg_l = np.mean(image[:, :, 0])
    image_std_l = np.std(image[:, :, 0])
    image_avg_a = np.mean(image[:, :, 1])
    image_std_a = np.std(image[:, :, 1])
    image_avg_b = np.mean(image[:, :, 2])
    image_std_b = np.std(image[:, :, 2])
    avg.append(image_avg_l)
    avg.append(image_avg_a)
    avg.append(image_avg_b)
    std.append(image_std_l)
    std.append(image_std_a)
    std.append(image_std_b)
    return (avg, std)


if __name__ == "__main__":

    """
        Compute stain statistics for the dataset of interest, considering only the training portion (partition[0]) and saves statistics into json file 
        in the dedicated folder "datasets/extra/HE/Augmentation"

        python generate_randstain_statistics.py --dataset_name cytodark0 --path_dataset "cytoark/datasets/cytodark0/40x/256x256/image" --color_space "LAB"
    """
    parser = argparse.ArgumentParser(description="norm&jitter dataset lab statistics")
    parser.add_argument("--path_dataset", type=str, metavar="DIR", help="path to dataset")
    parser.add_argument(
        "--dataset_name", type=str, default="", metavar="DIR", help="dataset name"
    )
    parser.add_argument(
        "--color_space",
        type=str,
        default="LAB",
        choices=["LAB", "HED", "HSV"],
        help="dataset statistics color space",
    )
    args = parser.parse_args()

    color_space = args.color_space
    dataset_name = args.dataset_name
    path_dataset = args.path_dataset

    if dataset_name == "cytodark0":
        images_list = pd.read_csv(
            os.path.join(os.path.split(path_dataset)[0], "folds.csv")
        )
        partition = (
            images_list.groupby("fold")["img_id"]
            .apply(lambda x: x.values.tolist())
            .to_dict()
        )
        image_list = [img_id + ".png" for img_id in partition[0]]
    elif dataset_name == "Conic":
        images_list = pd.read_csv(
            os.path.join(os.path.split(path_dataset)[0], "folds_dedup.csv")
        )
        partition = (
            images_list.groupby("fold")["img_id"]
            .apply(lambda x: x.values.tolist())
            .to_dict()
        )
        image_list = [img_id + ".png" for img_id in partition[0]]
    elif dataset_name == "Pannuke":
        images_list = pd.read_csv(
            os.path.join(os.path.split(path_dataset)[0], "folds_dedup.csv")
        )
        partition = (
            images_list.groupby("fold")["img_id"]
            .apply(lambda x: x.values.tolist())
            .to_dict()
        )
        image_list = [img_id + ".png" for img_id in partition[0]]

    labL_avg_List = []
    labA_avg_List = []
    labB_avg_List = []
    labL_std_List = []
    labA_std_List = []
    labB_std_List = []

    t1 = time.time()
    i = 0

    # for class_dir in os.listdir(path_dataset):
    #     path_class = os.path.join(path_dataset, class_dir)
    #     print(path_class)

    #     path_class_list = os.listdir(path_class)
    #     if args.random:
    random.shuffle(image_list)

    for img_id in tqdm(image_list):
        # path_img = os.path.join(path_class, image)
        # img = cv2.imread(path_img)
        try:  # debug
            if color_space == "LAB":
                img = cv2.cvtColor(
                    cv2.imread(
                        os.path.join(path_dataset, "{0}".format(img_id)), cv2.IMREAD_COLOR
                    ),
                    cv2.COLOR_BGR2LAB,
                )
                # cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            elif color_space == "HED":
                img = cv2.cvtColor(
                    cv2.imread(
                        os.path.join(path_dataset, "{0}".format(img_id)), cv2.IMREAD_COLOR
                    ),
                    cv2.COLOR_BGR2RGB,
                )
                img = color.rgb2hed(img)
            elif color_space == "HSV":
                img = cv2.cvtColor(
                    cv2.imread(
                        os.path.join(path_dataset, "{0}".format(img_id)), cv2.IMREAD_COLOR
                    ),
                    cv2.COLOR_BGR2HSV,
                )
            else:
                print("wrong color space: {}!!".format(color_space))
            img_avg, img_std = getavgstd(img)
        except:
            print(os.path.join(path_dataset, "{0}.png".format(img_id)))
            continue

        labL_avg_List.append(img_avg[0])
        labA_avg_List.append(img_avg[1])
        labB_avg_List.append(img_avg[2])
        labL_std_List.append(img_std[0])
        labA_std_List.append(img_std[1])
        labB_std_List.append(img_std[2])

    t2 = time.time()
    print(t2 - t1)
    l_avg_mean = np.mean(labL_avg_List).item()
    l_avg_std = np.std(labL_avg_List).item()
    l_std_mean = np.mean(labL_std_List).item()
    l_std_std = np.std(labL_std_List).item()
    a_avg_mean = np.mean(labA_avg_List).item()
    a_avg_std = np.std(labA_avg_List).item()
    a_std_mean = np.mean(labA_std_List).item()
    a_std_std = np.std(labA_std_List).item()
    b_avg_mean = np.mean(labB_avg_List).item()
    b_avg_std = np.std(labB_avg_List).item()
    b_std_mean = np.mean(labB_std_List).item()
    b_std_std = np.std(labB_std_List).item()

    std_avg_list = [
        labL_avg_List,
        labL_std_List,
        labA_avg_List,
        labA_std_List,
        labB_avg_List,
        labB_std_List,
    ]
    distribution = []
    for std_avg in std_avg_list:
        f = Fitter(std_avg, distributions=["norm", "laplace"])
        f.fit()
        distribution.append(list(f.get_best(method="sumsquare_error").keys())[0])

    yaml_dict_lab = {
        "random": True,
        "n_each_class": 0,
        "color_space": "LAB",
        "methods": "",
        "{}".format(color_space[0]): {  # lab-L/hed-H
            "avg": {
                "mean": round(l_avg_mean, 3),
                "std": round(l_avg_std, 3),
                "distribution": distribution[0],
            },
            "std": {
                "mean": round(l_std_mean, 3),
                "std": round(l_std_std, 3),
                "distribution": distribution[1],
            },
        },
        "{}".format(color_space[1]): {  # lab-A/hed-E
            "avg": {
                "mean": round(a_avg_mean, 3),
                "std": round(a_avg_std, 3),
                "distribution": distribution[2],
            },
            "std": {
                "mean": round(a_std_mean, 3),
                "std": round(a_std_std, 3),
                "distribution": distribution[3],
            },
        },
        "{}".format(color_space[2]): {  # lab-B/hed-D
            "avg": {
                "mean": round(b_avg_mean, 3),
                "std": round(b_avg_std, 3),
                "distribution": distribution[4],
            },
            "std": {
                "mean": round(b_std_mean, 3),
                "std": round(b_std_std, 3),
                "distribution": distribution[5],
            },
        },
    }

    print(os.path.dirname(os.path.dirname(os.getcwd())))

    yaml_save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.getcwd())),
        "datasets/extra/HE/Augmentation",
        "{}/{}TEST.yaml".format(
            "./",
            dataset_name
            if dataset_name != ""
            else "dataset_{}_random{}_n{}".format(color_space, True, 0),
        ),
    )
    with open(yaml_save_path, "w") as f:
        yaml.dump(yaml_dict_lab, f)
        print("The dataset lab statistics has been saved in {}".format(yaml_save_path))


####### Padova #######
# color_space = "LAB"
# dataset_name = "cytodark0"
# path_dataset = "D:/NucleiSegmentation/Projects/cytoark/datasets/cytodark0/40x/256x256/image"
# # images_list = glob.glob(path_dataset+"/*/*") # all files absolute path

####### Conic #######
# color_space = "LAB"
# dataset_name = "Conic" #
# path_dataset = "D:\NucleiSegmentation\Projects\cytoark\datasets\Conic\20x\256x256\image"

####### Pannuke #######
# color_space = "LAB"
# dataset_name = "Pannuke"
# path_dataset = "D:\NucleiSegmentation\Projects\cytoark\datasets\Pannuke\40x\256x256\image"

# file_set = set()

# for dir_, _, files in os.walk(path_dataset):
#     for file_name in files:
#         rel_dir = os.path.relpath(dir_, path_dataset)
#         rel_file = os.path.join(rel_dir, file_name)
#         file_set.add(rel_file)

# image_list = list(file_set)

####### Picasso #######
# color_space = "LAB"
# dataset_name = "Picasso"
# path_dataset = "E:/PicassoHistology/patch_neutrophils/patch_neutrophils/images"
# file_set = set()

# for dir_, _, files in os.walk(path_dataset):
#     for file_name in files:
#         rel_dir = os.path.relpath(dir_, path_dataset)
#         rel_file = os.path.join(rel_dir, file_name)
#         file_set.add(rel_file)

# image_list = list(file_set)