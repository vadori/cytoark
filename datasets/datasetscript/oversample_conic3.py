import pandas as pd
import cv2
import os
import numpy as np
import skimage
import argparse
from tqdm import tqdm
 
CLASS_NAMES = {
    0: "BACKGROUND",
    1: "Neutrophil",
    2: "Epithelial",
    3: "Lymphocyte",
    4: "Plasma",
    5: "Eosinophil",
    6: "Connective",
}

SEED = 5

parser = argparse.ArgumentParser(description="Oversampling CONIC datasey",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-df", "--data_folder", help="Path to the data for training")
parser.add_argument("-vdf", "--valid_data_folder", help="Path to the data for validation")
#parser.add_argument("-exp", "--exponential_factor", type=float, default = 0.75, help="Exponential factor to control the amount of augmentation")
args = vars(parser.parse_args())

# Set up parameters
data_folder = args["data_folder"]
valid_data_folder = args["valid_data_folder"]
np.random.seed(SEED)
#exponential_factor = args["exponential_factor"]

image_folder = os.path.join(data_folder, 'image')
instance_mask_folder = os.path.join(data_folder, 'label')
class_mask_folder = os.path.join(data_folder, 'class')
valid_image_folder = os.path.join(valid_data_folder, 'image')
valid_instance_mask_folder = os.path.join(valid_data_folder, 'label')
valid_class_mask_folder = os.path.join(valid_data_folder, 'class')

df = pd.read_csv(os.path.join(data_folder, 'folds_dedup.csv'))
partition = df.groupby('fold')['img_id'].apply(lambda x: x.values.tolist()).to_dict()

# TRAIN
counts = np.empty((0,7), int)
for img_id in tqdm(partition[0]):
    img = cv2.cvtColor(cv2.imread(os.path.join(image_folder, "{0}.png".format(img_id)), cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)  
    labelmap = cv2.imread(os.path.join(instance_mask_folder, "{0}.tiff".format(img_id)), -1)  
    cell_class_mask = cv2.imread(os.path.join(class_mask_folder, "{0}.tiff".format(img_id)), -1) 
    rp = skimage.measure.regionprops(labelmap, intensity_image=cell_class_mask)
    count = np.bincount([int(rp[i].intensity_max) for i in np.arange(len(rp))], minlength=7).reshape((1,7))
    counts = np.append(counts, count, axis=0)

df.index = df['img_id']
dftrain = df.loc[partition[0]]
dftrain = pd.concat([dftrain, pd.DataFrame(counts, index=dftrain.index)], axis=1).drop(columns="img_id").reset_index()

class_counts = np.sum(counts,axis = 0)
extra_classes = np.argsort(class_counts)[1:-1]
n_extras = np.sqrt(np.sum(class_counts) / class_counts[extra_classes])
n_extras = (n_extras / np.max(n_extras) * len(partition[0])).astype(int)
n_extras = (n_extras * [1.6,1.6,0.9,0.9,0.9]).astype(int)

idx_take = np.empty((0,1), int)

for c, n_extra in zip(extra_classes, n_extras):

    dftrain["prob"] = dftrain[c]
    dftrain["prob"] = dftrain["prob"].fillna(0)
    prob = np.clip(dftrain["prob"].values, 0, np.percentile(dftrain["prob"].values, 97)) # was 99.8
    #prob = prob ** 2
    prob = prob / np.sum(prob)
    print(f"adding {n_extra} images of class {c} ({CLASS_NAMES[c]}) to the training set")
    idx_extra = np.random.choice(np.arange(len(dftrain)), n_extra, p=prob)
    idx_take = np.append(idx_take, idx_extra)

dftraincount = pd.concat([dftrain,dftrain.iloc[idx_take]]).sample(frac=1).reset_index(drop=True)
dftrain = dftraincount[["img_id","fold"]]

# # VALID
counts = np.empty((0,7), int)
for img_id in tqdm(partition[1]):
    img = cv2.cvtColor(cv2.imread(os.path.join(valid_image_folder, "{0}.png".format(img_id)), cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)  
    labelmap = cv2.imread(os.path.join(valid_instance_mask_folder, "{0}.tiff".format(img_id)), -1)  
    cell_class_mask = cv2.imread(os.path.join(valid_class_mask_folder, "{0}.tiff".format(img_id)), -1) 
    rp = skimage.measure.regionprops(labelmap, intensity_image=cell_class_mask)
    count = np.bincount([int(rp[i].intensity_max) for i in np.arange(len(rp))], minlength=7).reshape((1,7))
    counts = np.append(counts, count, axis=0)

dfvalid = df.loc[partition[1]]
# to be changed to dfvalid if valid gets upsampled 
dfvalidcount = pd.concat([dfvalid, pd.DataFrame(counts, index=dfvalid.index)], axis=1).drop(columns="img_id").reset_index()

# class_counts = np.sum(counts,axis = 0)
# extra_classes = np.argsort(class_counts)[1:-1]
# n_extras = np.sqrt(np.sum(class_counts) / class_counts[extra_classes])*2
# n_extras = (n_extras / np.max(n_extras) * len(partition[1])).astype(int)
# n_extras = n_extras * [2,2,1,1,1]

# idx_take = np.empty((0,1), int)

# for c, n_extra in zip(extra_classes, n_extras):

#     dfvalid["prob"] = dfvalid[c]
#     dfvalid["prob"] = dfvalid["prob"].fillna(0)
#     prob = np.clip(dfvalid["prob"].values, 0, np.percentile(dfvalid["prob"].values, 99.8))
#     #prob = prob**2
#     prob = prob / np.sum(prob)
#     print(f"adding {n_extra} images of class {c} ({CLASS_NAMES[c]}) to the validation set")
#     idx_extra = np.random.choice(np.arange(len(dfvalid)), n_extra, p=prob)
#     idx_take = np.append(idx_take, idx_extra)

# dfvalidcount = pd.concat([dfvalid,dfvalid.iloc[idx_take]]).sample(frac=1).reset_index(drop=True)
# dfvalid = dfvalidcount[["img_id","fold"]]

# # TEST (only count)
counts = np.empty((0,7), int)
for img_id in tqdm(partition[2]):
    img = cv2.cvtColor(cv2.imread(os.path.join(valid_image_folder, "{0}.png".format(img_id)), cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)  
    labelmap = cv2.imread(os.path.join(valid_instance_mask_folder, "{0}.tiff".format(img_id)), -1)  
    cell_class_mask = cv2.imread(os.path.join(valid_class_mask_folder, "{0}.tiff".format(img_id)), -1) 
    rp = skimage.measure.regionprops(labelmap, intensity_image=cell_class_mask)
    count = np.bincount([int(rp[i].intensity_max) for i in np.arange(len(rp))], minlength=7).reshape((1,7))
    counts = np.append(counts, count, axis=0)

dftest = df.loc[partition[2]]
dftestcount = pd.concat([dftest, pd.DataFrame(counts, index=dftest.index)], axis=1).drop(columns="img_id").reset_index()


dfcount = pd.concat([dftraincount,dfvalidcount,dftestcount])
dfcount.to_csv(os.path.join(data_folder,'folds_counts_withduplicates.csv'), index=False)
dfcount.drop_duplicates().to_csv(os.path.join(data_folder,'folds_counts.csv'), index=False)

#df = pd.concat([dftrain,df.loc[partition[1]],df.loc[partition[2]]])
df = pd.concat([dftrain,dfvalid,dftest])
df.to_csv(os.path.join(data_folder,'folds.csv'), index=False)

partitionnew = df.groupby('fold')['img_id'].apply(lambda x: x.values.tolist()).to_dict()

## COUNTING 

# TRAIN
counts = np.empty((0,7), int)
for img_id in tqdm(partitionnew[0]):
    img = cv2.cvtColor(cv2.imread(os.path.join(image_folder, "{0}.png".format(img_id)), cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)  
    labelmap = cv2.imread(os.path.join(instance_mask_folder, "{0}.tiff".format(img_id)), -1)  
    cell_class_mask = cv2.imread(os.path.join(class_mask_folder, "{0}.tiff".format(img_id)), -1) 
    rp = skimage.measure.regionprops(labelmap, intensity_image=cell_class_mask)
    count = np.bincount([int(rp[i].intensity_max) for i in np.arange(len(rp))], minlength=7).reshape((1,7))
    counts = np.append(counts, count, axis=0)

class_counts = np.sum(counts,axis = 0)
for c in np.arange(7):
    print(f"Using {class_counts[c]} cells of class {c} ({CLASS_NAMES[c]}) for training")
df = pd.DataFrame(class_counts, index=CLASS_NAMES.values(), columns=["counts"])
df = df.drop("BACKGROUND")
df["%"] = (100 * (df["counts"] / df["counts"].sum())).round(2)
print(df) 

# VALID
counts = np.empty((0,7), int)
for img_id in tqdm(partitionnew[1]):
    img = cv2.cvtColor(cv2.imread(os.path.join(image_folder, "{0}.png".format(img_id)), cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)  
    labelmap = cv2.imread(os.path.join(instance_mask_folder, "{0}.tiff".format(img_id)), -1)  
    cell_class_mask = cv2.imread(os.path.join(class_mask_folder, "{0}.tiff".format(img_id)), -1) 
    rp = skimage.measure.regionprops(labelmap, intensity_image=cell_class_mask)
    count = np.bincount([int(rp[i].intensity_max) for i in np.arange(len(rp))], minlength=7).reshape((1,7))
    counts = np.append(counts, count, axis=0)

class_counts = np.sum(counts,axis = 0)
for c in np.arange(7):
    print(f"Using {class_counts[c]} cells of class {c} ({CLASS_NAMES[c]}) for validation")
df = pd.DataFrame(class_counts, index=CLASS_NAMES.values(), columns=["counts"])
df = df.drop("BACKGROUND")
df["%"] = (100 * (df["counts"] / df["counts"].sum())).round(2)
print(df) 

# TEST
counts = np.empty((0,7), int)
for img_id in tqdm(partitionnew[2]):
    img = cv2.cvtColor(cv2.imread(os.path.join(image_folder, "{0}.png".format(img_id)), cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)  
    labelmap = cv2.imread(os.path.join(instance_mask_folder, "{0}.tiff".format(img_id)), -1)  
    cell_class_mask = cv2.imread(os.path.join(class_mask_folder, "{0}.tiff".format(img_id)), -1) 
    rp = skimage.measure.regionprops(labelmap, intensity_image=cell_class_mask)
    count = np.bincount([int(rp[i].intensity_max) for i in np.arange(len(rp))], minlength=7).reshape((1,7))
    counts = np.append(counts, count, axis=0)

class_counts = np.sum(counts,axis = 0)
for c in np.arange(7):
    print(f"Using {class_counts[c]} cells of class {c} ({CLASS_NAMES[c]}) for testing")
df = pd.DataFrame(class_counts, index=CLASS_NAMES.values(), columns=["counts"])
df = df.drop("BACKGROUND")
df["%"] = (100 * (df["counts"] / df["counts"].sum())).round(2)
print(df) 

# python oversample_conic3.py -df "D:\NucleiSegmentation\Projects\cytoark\datasets\conic\20x\256x256" -vdf "D:\NucleiSegmentation\Projects\cytoark\datasets\conic\20x\256x256"