import os
import glob
import cv2
import sys

import numpy as np
from skimage.measure import label
from skimage.segmentation import watershed

# Get the parent directory by going one level up
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(parent_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)


from imutils import dir_distance_map, dir_distance_map_fullrange, distance_map, get_sobel_kernel

def generate_contig_labels(bwmask_folder, instance_mask_folder):

    for image in glob.glob(os.path.join(bwmask_folder, '*.png')):    

        gt = cv2.imread(image)

        gtlabel = label(gt)
        gtgray = np.zeros(gt.shape[:2], dtype=np.uint8)
        inst_list = list(np.unique(gtlabel))
        inst_list.remove(0) # 0 is background

        kernel = np.ones((3, 3), np.uint8)
        
        for inst_id in inst_list:
            inst_map = np.array(gtlabel == inst_id, np.uint8)[...,0]
            inst_dil = cv2.dilate(inst_map, kernel, iterations=1)
            gtgray = gtgray + inst_dil
            
        gtgraybord = np.array(gtgray > 1, np.uint8)
        newgt = watershed(255-gt[...,0], label(gt[...,0]), mask=gtgraybord + 1/255*gt[...,0], watershed_line=False)
        img = label(newgt)
        
        cv2.imwrite(os.path.join(instance_mask_folder, image.split('.p')[0].split('\\')[-1]+'.tiff'), img)

def generate_dir_distance_maps(instance_mask_folder, dir_distance_map_folder, magnification = 40, fullrange = False, relabel = False):
    for image in glob.glob(os.path.join(instance_mask_folder, '*.tiff')):    
            print("Processing image: ", image)
            gt = cv2.imread(image,-1)
            if relabel:
                gt = label(gt,connectivity=1)
                cv2.imwrite(image, gt)
            #plt.imshow(gt)
            if fullrange:
                 h_map, v_map, tl_map, bl_map = dir_distance_map_fullrange(gt)
            else:
                h_map, v_map, tl_map, bl_map = dir_distance_map(gt, magnification)
            #plt.figure(figsize=(12,8), dpi= 600, facecolor='w', edgecolor='k')
            #plt.imshow(h_map)
            cv2.imwrite(os.path.join(dir_distance_map_folder, image.split('.t')[0].split('\\')[-1]+'_h.tiff'), h_map)
            cv2.imwrite(os.path.join(dir_distance_map_folder, image.split('.t')[0].split('\\')[-1]+'_v.tiff'), v_map)
            cv2.imwrite(os.path.join(dir_distance_map_folder, image.split('.t')[0].split('\\')[-1]+'_tl.tiff'), tl_map)
            cv2.imwrite(os.path.join(dir_distance_map_folder, image.split('.t')[0].split('\\')[-1]+'_bl.tiff'), bl_map)

def generate_distance_maps(instance_mask_folder, dir_distance_map_folder, distance_map_folder, magnification = 40, relabel = False):
        for image in glob.glob(os.path.join(instance_mask_folder, '*.tiff')):    
            print("Processing image: ", image)
            gt = (cv2.imread(image,-1))

            h_dir = cv2.imread(os.path.join(dir_distance_map_folder, image.split('.t')[0].split('\\')[-1]+'_h.tiff'),-1)
            v_dir = cv2.imread(os.path.join(dir_distance_map_folder, image.split('.t')[0].split('\\')[-1]+'_v.tiff'),-1)
            tl_dir = cv2.imread(os.path.join(dir_distance_map_folder, image.split('.t')[0].split('\\')[-1]+'_tl.tiff'),-1)
            bl_dir = cv2.imread(os.path.join(dir_distance_map_folder, image.split('.t')[0].split('\\')[-1]+'_bl.tiff'),-1)

            dist = distance_map(h_dir,v_dir,tl_dir,bl_dir,gt)
            cv2.imwrite(os.path.join(distance_map_folder, image.split('.t')[0].split('\\')[-1]+'.tiff'), dist)

if __name__ == "__main__":

    # magnifications = [20] 
    # dataset = "Lilly" 
    # relabel = True

    # magnifications = [20] 
    # dataset = "Picasso" 
    # relabel = True

    # if dataset in ["Conic", "Picasso"]:
    #     magnifications = [20] 
    # elif dataset == "Pannuke":
    #     magnifications = [40]

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

    # magnifications = [40] 
    # dataset = "cytoark"
    # relabel = True

    magnifications = [40] 
    dataset = "cytoarkNEW"
    relabel = True
    
    # rootfolder = os.path.join("D:/NucleiSegmentation/Projects/CISCA/NeuronInstanceSeg-master/NeuronInstanceSeg-master/datasets/Conic",str(magnification)+"x")
    
    for magnification in magnifications:
        
        rootfolder = os.path.join("D:/NucleiSegmentation/Projects/cytoark/datasets",dataset,str(magnification)+"x")
    
        if dataset == "Conic":
            sizes = [256]
            relabel = False
        elif dataset == "Pannuke" or dataset == "Picasso" or dataset == "Lilly":
            sizes = [256]
        elif dataset == "cytoark":
            if magnification == 20: 
                sizes = [256,512,1024]
            else:
                sizes = [256,512,1024,2048]
        elif dataset == "cytoarkNEW":
            if magnification == 20: 
                sizes = [256,512,1024]
            else:
                sizes = [256,512,1024,2048]
        elif magnification == 20: #Padova
            sizes = [256,270,370,512,1024]
        else:
            sizes = [256,270,370,512,1024,2048]

        fullrange = False
        for size in sizes:
            sizestring = str(size)+"x"+str(size)
            instance_mask_folder=os.path.join(rootfolder,sizestring, "label")
            distance_map_folder=os.path.join(rootfolder,sizestring, "dist")
            if fullrange:
                dir_distance_map_folder=os.path.join(rootfolder,sizestring, "distmapfull")
            else:
                dir_distance_map_folder=os.path.join(rootfolder,sizestring, "distmap")
            generate_dir_distance_maps(instance_mask_folder, dir_distance_map_folder,magnification=magnification,fullrange=fullrange, relabel=relabel)
            generate_distance_maps(instance_mask_folder, dir_distance_map_folder,distance_map_folder, magnification = magnification, relabel = False)

        fullrange = True
        for size in sizes:
            sizestring = str(size)+"x"+str(size)
            instance_mask_folder=os.path.join(rootfolder,sizestring, "label")
            if fullrange:
                dir_distance_map_folder=os.path.join(rootfolder,sizestring, "distmapfull")
            else:
                dir_distance_map_folder=os.path.join(rootfolder,sizestring, "distmap")
            generate_dir_distance_maps(instance_mask_folder, dir_distance_map_folder,magnification=magnification,fullrange=fullrange, relabel=False)
            
# TO DO: delete!
        
# test distance
        
# import os
# import cv2
# import glob
# import numpy as np
# from skimage import morphology as morph
# from skimage.measure import label
# from scipy import ndimage
# from skimage.segmentation import watershed
# import matplotlib.pyplot as plt

# from mask_transform_utils import distance_map

# magnification = "20x"
# size = "256x256"
# rootfolder = "D:/NucleiSegmentation/Projects/CISCA/NeuronInstanceSeg-master/NeuronInstanceSeg-master/datasets/Padova/" + magnification
 
# instance_mask_folder=os.path.join(rootfolder,size,"label")
# dir_distance_map_folder=os.path.join(rootfolder,size,"distmap")

# image = glob.glob(os.path.join(instance_mask_folder, '*.tiff'))[1:2][0]    
# print("Processing image: ", image)
# gt = (cv2.imread(image,-1))

# h_dir = cv2.imread(os.path.join(dir_distance_map_folder, image.split('.t')[0].split('\\')[-1]+'_h.tiff'),-1)
# v_dir = cv2.imread(os.path.join(dir_distance_map_folder, image.split('.t')[0].split('\\')[-1]+'_v.tiff'),-1)
# tl_dir = cv2.imread(os.path.join(dir_distance_map_folder, image.split('.t')[0].split('\\')[-1]+'_tl.tiff'),-1)
# bl_dir = cv2.imread(os.path.join(dir_distance_map_folder, image.split('.t')[0].split('\\')[-1]+'_bl.tiff'),-1)

# dist = distance_map(h_dir,v_dir,tl_dir,bl_dir,gt)

# import matplotlib.pyplot as plt
# plt.imshow(dist)
# plt.colorbar()

#     # to be saved as separate script to generate labels with watershed_line=False from binary gt image 
# from skimage.measure import label
# import numpy as np
# import os
# import glob
# import cv2

# def generate_contig_labels(bwmask_folder, instance_mask_folder):

#     for image in glob.glob(os.path.join(bwmask_folder, '*.png')):    

#         gt = cv2.imread(image)

#         gtlabel = label(gt)
#         gtgray = np.zeros(gt.shape[:2], dtype=np.uint8)
#         inst_list = list(np.unique(gtlabel))
#         inst_list.remove(0) # 0 is background

#         kernel = np.ones((3, 3), np.uint8)
        
#         for inst_id in inst_list:
#             inst_map = np.array(gtlabel == inst_id, np.uint8)[...,0]
#             inst_dil = cv2.dilate(inst_map, kernel, iterations=1)
#             gtgray = gtgray + inst_dil
            
#         gtgraybord = np.array(gtgray > 1, np.uint8)
#         newgt = watershed(255-gt[...,0], label(gt[...,0]), mask=gtgraybord + 1/255*gt[...,0], watershed_line=False)
#         img = label(newgt)
        
#         cv2.imwrite(os.path.join(instance_mask_folder, image.split('.p')[0].split('\\')[-1]+'.tiff'), img)

# # to be saved as separate script to generate gradient maps
# def generate_distance_maps(instance_mask_folder, dir_distance_map_folder):
#     for image in glob.glob(os.path.join(instance_mask_folder, '*.tiff')):    
#             print(image)
#             gt = cv2.imread(image,-1)
#             #plt.imshow(gt)
#             h_map, v_map, tl_map, bl_map = hover_mask3(gt)
#             #plt.figure(figsize=(12,8), dpi= 600, facecolor='w', edgecolor='k')
#             #plt.imshow(h_map)
#             cv2.imwrite(os.path.join(dir_distance_map_folder, image.split('.t')[0].split('\\')[-1]+'_h.tiff'), h_map)
#             cv2.imwrite(os.path.join(dir_distance_map_folder, image.split('.t')[0].split('\\')[-1]+'_v.tiff'), v_map)
#             cv2.imwrite(os.path.join(dir_distance_map_folder, image.split('.t')[0].split('\\')[-1]+'_tl.tiff'), tl_map)
#             cv2.imwrite(os.path.join(dir_distance_map_folder, image.split('.t')[0].split('\\')[-1]+'_bl.tiff'), bl_map)

# # # with the new mode for label generation, no need to generate label iamges here as they are created in matlab
# # # BWMASK_FOLDER = os.path.join(rootfolder,"2048x2048\\bwmask"
# # # instance_mask_folder = os.path.join(rootfolder,"2048x2048\\label"
# # # generate_contig_labels(BWMASK_FOLDER, instance_mask_folder)
# # # BWMASK_FOLDER = os.path.join(rootfolder,"256x256\\bwmask"
# # # instance_mask_folder = os.path.join(rootfolder,"256x256\\label"
# # # generate_contig_labels(BWMASK_FOLDER, instance_mask_folder)
# # # BWMASK_FOLDER = os.path.join(rootfolder,"270x270\\bwmask"
# # # instance_mask_folder = os.path.join(rootfolder,"270x270\\label"
# # # generate_contig_labels(BWMASK_FOLDER, instance_mask_folder)
