import os
import glob
import cv2
import sys
import pathlib

import numpy as np
from skimage.measure import label
from skimage.segmentation import watershed

# Get the parent directory by going one level up
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(parent_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)


from imutils import (
    dir_distance_map,
    dir_distance_map_fullrange,
    distance_map,
    get_sobel_kernel,
)


def generate_contig_labels(bwmask_folder, instance_mask_folder):
    for image in glob.glob(os.path.join(bwmask_folder, "*.png")):
        gt = cv2.imread(image)

        gtlabel = label(gt)
        gtgray = np.zeros(gt.shape[:2], dtype=np.uint8)
        inst_list = list(np.unique(gtlabel))
        inst_list.remove(0)  # 0 is background

        kernel = np.ones((3, 3), np.uint8)

        for inst_id in inst_list:
            inst_map = np.array(gtlabel == inst_id, np.uint8)[..., 0]
            inst_dil = cv2.dilate(inst_map, kernel, iterations=1)
            gtgray = gtgray + inst_dil

        gtgraybord = np.array(gtgray > 1, np.uint8)
        newgt = watershed(
            255 - gt[..., 0],
            label(gt[..., 0]),
            mask=gtgraybord + 1 / 255 * gt[..., 0],
            watershed_line=False,
        )
        img = label(newgt)

        cv2.imwrite(
            os.path.join(
                instance_mask_folder, image.split(".p")[0].split("\\")[-1] + ".tiff"
            ),
            img,
        )


def generate_dir_distance_maps(
    instance_mask_folder,
    dir_distance_map_folder,
    magnification=40,
    fullrange=False,
    relabel=False,
):
    for image in glob.glob(os.path.join(instance_mask_folder, "*.tiff")):
        print("Processing image: ", image)
        gt = cv2.imread(image, -1)
        if relabel:
            gt = label(gt, connectivity=1)
            cv2.imwrite(image, gt)
        # plt.imshow(gt)
        if fullrange:
            h_map, v_map, tl_map, bl_map = dir_distance_map_fullrange(gt)
        else:
            h_map, v_map, tl_map, bl_map = dir_distance_map(gt, magnification)
        # plt.figure(figsize=(12,8), dpi= 600, facecolor='w', edgecolor='k')
        # plt.imshow(h_map)
        cv2.imwrite(
            os.path.join(
                dir_distance_map_folder,
                image.split(".t")[0].split("\\")[-1] + "_h.tiff",
            ),
            h_map,
        )
        cv2.imwrite(
            os.path.join(
                dir_distance_map_folder,
                image.split(".t")[0].split("\\")[-1] + "_v.tiff",
            ),
            v_map,
        )
        cv2.imwrite(
            os.path.join(
                dir_distance_map_folder,
                image.split(".t")[0].split("\\")[-1] + "_tl.tiff",
            ),
            tl_map,
        )
        cv2.imwrite(
            os.path.join(
                dir_distance_map_folder,
                image.split(".t")[0].split("\\")[-1] + "_bl.tiff",
            ),
            bl_map,
        )


def generate_distance_maps(
    instance_mask_folder,
    dir_distance_map_folder,
    distance_map_folder,
    magnification=40,
    relabel=False,
):
    for image in glob.glob(os.path.join(instance_mask_folder, "*.tiff")):
        print("Processing image: ", image)
        gt = cv2.imread(image, -1)

        h_dir = cv2.imread(
            os.path.join(
                dir_distance_map_folder,
                image.split(".t")[0].split("\\")[-1] + "_h.tiff",
            ),
            -1,
        )
        v_dir = cv2.imread(
            os.path.join(
                dir_distance_map_folder,
                image.split(".t")[0].split("\\")[-1] + "_v.tiff",
            ),
            -1,
        )
        tl_dir = cv2.imread(
            os.path.join(
                dir_distance_map_folder,
                image.split(".t")[0].split("\\")[-1] + "_tl.tiff",
            ),
            -1,
        )
        bl_dir = cv2.imread(
            os.path.join(
                dir_distance_map_folder,
                image.split(".t")[0].split("\\")[-1] + "_bl.tiff",
            ),
            -1,
        )

        dist = distance_map(h_dir, v_dir, tl_dir, bl_dir, gt)
        cv2.imwrite(
            os.path.join(
                distance_map_folder, image.split(".t")[0].split("\\")[-1] + ".tiff"
            ),
            dist,
        )


if __name__ == "__main__":

    # magnifications = [20]
    # dataset = "Conic"
    # relabel = True

    # magnifications = [40]
    # dataset = "Pannuke"
    # relabel = False

    # magnifications = [20,40]
    # dataset = "Padova"
    # relabel = False

    # magnifications = [20]
    # dataset = "Padova"
    # relabel = True

    magnifications = [40]
    dataset = "cytodark0"
    relabel = True

    for magnification in magnifications:
        rootfolder = os.path.join(
            os.path.join(pathlib.Path(__file__).parents[1],"datasets"),
            dataset,
            str(magnification) + "x",
        )

        if dataset == "Conic":
            sizes = [256]
            relabel = False
        elif dataset == "Pannuke":
            sizes = [256]
        elif dataset == "cytodark0":
            if magnification == 20:
                sizes = [256, 512, 1024]
            else:
                sizes = [256, 512, 1024, 2048]
        elif magnification == 20:  # Padova
            sizes = [256, 270, 370, 512, 1024]
        else:
            sizes = [256, 270, 370, 512, 1024, 2048]

        fullrange = False
        for size in sizes:
            sizestring = str(size) + "x" + str(size)
            instance_mask_folder = os.path.join(rootfolder, sizestring, "label")
            # distance_map_folder = os.path.join(rootfolder, sizestring, "dist")
            if fullrange:
                dir_distance_map_folder = os.path.join(
                    rootfolder, sizestring, "distmap"
                )
            else:
                dir_distance_map_folder = os.path.join(
                    rootfolder, sizestring, "distmaptips"
                )
            generate_dir_distance_maps(
                instance_mask_folder,
                dir_distance_map_folder,
                magnification=magnification,
                fullrange=fullrange,
                relabel=relabel,
            )
            # generate_distance_maps(
            #     instance_mask_folder,
            #     dir_distance_map_folder,
            #     distance_map_folder,
            #     magnification=magnification,
            #     relabel=False,
            # )

        fullrange = True
        for size in sizes:
            sizestring = str(size) + "x" + str(size)
            instance_mask_folder = os.path.join(rootfolder, sizestring, "label")
            if fullrange:
                dir_distance_map_folder = os.path.join(
                    rootfolder, sizestring, "distmap"
                )
            else:
                dir_distance_map_folder = os.path.join(
                    rootfolder, sizestring, "distmaptips"
                )
            generate_dir_distance_maps(
                instance_mask_folder,
                dir_distance_map_folder,
                magnification=magnification,
                fullrange=fullrange,
                relabel=False,
            )
