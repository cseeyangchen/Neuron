# Data Preparation

### Download datasets.

#### There are 3 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- NW-UCLA

#### NTU RGB+D 60 and 120

1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`

#### PKU MMD

1. Request and download the dataset [here](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html)
2. Unzip all skeleton files from `Skeleton.7z` to `./data/pkummd_raw/part1`
3. Unzip all label files from `Label_PKUMMD.7z` to `./data/pkummd_raw/part1`
3. Unzip all skeleton files from `Skeleton_v2.7z` to `./data/pkummd_raw/part2`
4. Unzip all label files from `Label_PKUMMD_v2.7z` to `./data/pkummd_raw/part2`


### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - NW-UCLA/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./data/ntu60 # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```

- Generate PKU MMD I or PKU MMD II dataset:
```
 cd ./data/pkummd/part1 # or cd ./data/pkummd/part2
 mkdir skeleton_pku_v1 or mkdir skeleton_pku_v2
 # Get skeleton of each performer
 python pku_part1_skeleton.py or python pku_part2_skeleton.py
 # Transform the skeleton to the center of the first frame
 python pku_part1_gendata.py or python pku_part2_gendata.py
 # Downsample the frame to 64
 python preprocess_pku.py
 # Concatenate train data and val data into one file
 python pku_concat.py
```



