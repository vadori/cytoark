# %%

import numpy as np
import joblib
import os
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit

SEED = 5
info = pd.read_csv('E:\Conic\CoNIC Challenge Train-20231214T165512Z-002\CoNIC Challenge Train\patch_info.csv')
file_names = np.squeeze(info.to_numpy()).tolist()

# %%
img_sources = [v.split('-')[0] for v in file_names]
img_sources = np.unique(img_sources)

cohort_sources = [v.split('_')[0] for v in img_sources]
_, cohort_sources = np.unique(cohort_sources, return_inverse=True)
# %%
num_trials = 10
splitter = StratifiedShuffleSplit(
    n_splits=num_trials,
    train_size=0.8,
    test_size=0.2,
    random_state=SEED
)

splits = []
split_generator = splitter.split(img_sources, cohort_sources)

for train_indices, test_indices in split_generator:
    train_cohorts = img_sources[train_indices]
    test_cohorts = img_sources[test_indices]
    assert np.intersect1d(train_cohorts, test_cohorts).size == 0
    train_names = [
        file_name
        for file_name in file_names
        for source in train_cohorts
        if source == file_name.split('-')[0]
    ]
    test_names = [
        file_name
        for file_name in file_names
        for source in test_cohorts
        if source == file_name.split('-')[0]
    ]
    train_names = np.unique(train_names)
    test_names = np.unique(test_names)
    print(f'Train: {len(train_names):04d} - Test: {len(test_names):04d}')
    assert np.intersect1d(train_names, test_names).size == 0
    train_indices = [file_names.index(v) for v in train_names]
    test_indices = [file_names.index(v) for v in test_names]
    splits.append({
        'train': train_indices,
        'test': test_indices
    })
joblib.dump(splits, 'splits.dat')

# %%
FOLD_IDX = 1

splits = joblib.load('splits.dat')
splits[FOLD_IDX]

traindata = [file_names[v] for v in splits[FOLD_IDX]["train"]]
test_names = [file_names[v] for v in splits[FOLD_IDX]["test"]]

num_trials = 1
splitter = StratifiedShuffleSplit(
    n_splits=num_trials,
    train_size=0.9,
    test_size=0.1,
    random_state=SEED
)

img_sources = [v.split('-')[0] for v in traindata]
img_sources = np.unique(img_sources)

cohort_sources = [v.split('_')[0] for v in img_sources]
_, cohort_sources = np.unique(cohort_sources, return_inverse=True)

split_generator = splitter.split(img_sources, cohort_sources)
for train_indices, valid_indices in split_generator:
    train_cohorts = img_sources[train_indices]
    valid_cohorts = img_sources[valid_indices]
    assert np.intersect1d(train_cohorts, valid_cohorts).size == 0
    train_names = [
        file_name
        for file_name in file_names
        for source in train_cohorts
        if source == file_name.split('-')[0]
    ]

    valid_names = [
        file_name
        for file_name in file_names
        for source in valid_cohorts
        if source == file_name.split('-')[0]
    ]

    train_names = np.unique(train_names)
    valid_names = np.unique(valid_names)
    print(f'Train: {len(train_names):04d} - Valid: {len(valid_names):04d}')
    assert np.intersect1d(train_names, valid_names).size == 0
    #train_indices = [file_names.index(v) for v in train_names]
    #valid_indices = [file_names.index(v) for v in valid_names]


d = {'img_id': list(train_names)+list(valid_names)+test_names, 
     'fold': [0]*len(train_names)+[1]*len(valid_names)+[2]*len(test_names)}
df = pd.DataFrame(data=d)

outputfolder = "D:/NucleiSegmentation/Projects/cytoark/datasets/conic/20x/256x256"
df.to_csv(os.path.join(outputfolder,'folds_dedup.csv'), index=False)
# %%
