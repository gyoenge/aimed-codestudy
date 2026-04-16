"""
Flow: 
    Intensity + Shape + Texture Features for each cells 
    (여기서는 cell마다 이미 계산이 되어있다고 가정) 
    --> aggregate within each patch, by calculating statistics (mean, std, skew, kurtosis)
    --> feature vector  
"""

import pickle 
import numpy as np 
from scipy.stats import skew, kurtosis 

### Input Assumption 

cell_properties_path = "path/to/cell_properties" 
cell_pickle_path = "path/to/cell_pickle" # 이미 HoVerNet 등으로 nucleus segmentation + per-cell feature 계산이 끝난 상태. 
with open(cell_properties_path + '/' + cell_pickle_path, 'rb') as f:
    cell_prop = pickle.load(f)
""" 
    각 세포마다 
        Intensity + Shape + Texture Features
""" 

# patch/cell settings 
hovernet_mag = 40 
patch_extract_mag = 5 
patch_size = 224 
no_of_cell_types = 5 # hovernet pannuke 
outlier_removal = 0.05 # removing outliers which could be caused by incorrect segmentations from hovernet 
use_index = True # since our patches are saved as row col 

# patch안에 있는 cell만 필터링 
def single_crop_features(
    key_list, 
    cell_centroid_list, 
    type_list, 
    property_list, 
    patch_name, 
    patch_mag_ratio, 
    height, 
    width, 
): 
    _, column, row = patch_name.split('/')[-1].split('.')[0].split('_') # patch_name: "xxx_10_20.png" 
    column = int(column) * patch_mag_ratio
    row = int(row) * patch_mag_ratio

    start_x_point, stop_x_point = column, column + width
    start_y_point, stop_y_point = row, row + height 
    """
    (x_start, y_start)
                ┌───────────┐
                │   patch   │
                └───────────┘
                        (x_end, y_end)
    """

    x_lower = np.where(cell_centroid_list[:,0] > start_x_point)[0].copy() # np.Array 
    x_upper = np.where(cell_centroid_list[:,0] < stop_x_point)[0].copy() 
    y_lower = np.where(cell_centroid_list[:,1] > start_y_point)[0].copy() 
    y_upper = np.where(cell_centroid_list[:,1] < stop_y_point)[0].copy() 

    # (x조건)∩(y조건) : patch 내부에 있는 cell index만 남김
    x_intersection = np.intersect1d(x_lower, x_upper) # x 조건 만족하는 index
    y_intersection = np.intersect1d(y_lower, y_upper) # y 조건 만족하는 index
    centroids_in_region = np.intersect1d(x_intersection, y_intersection).copy() 

    # 해당 cell 정보 가져오기
    points_type = type_list[centroids_in_region].copy() 
    points_properties = property_list[centroids_in_region].copy() 
    """
    points_type
        [cell1_type, cell2_type, ...]
    points_properties
        [
            [feature1, feature2, ...],  # cell1
            [feature1, feature2, ...],  # cell2
        ]
    """

    # cell staticstics class-wise ! 
    cell_statistics_class_wise = [] 
    
    # (1) add number of each type of cells 
    for i in range(1, 1 + no_of_cell_types): 
        cell_statistics_class_wise.append(
            np.where(points_type==i)[0].shape[0] 
        )
        # 각 row 마다 class type 개수가 추가됨. 
    
    # (2) add statistics of properties 
    for i in range(1, 1 + no_of_cell_types): 
        # 아무 것도 없는 class type에 대한 처리 
        if np.where(points_type==i)[0].shape[0] ==0: 
            cell_statistics_class_wise.extend([None] * property_list.shape[1]) # zeros mean 
            cell_statistics_class_wise.extend([None] * property_list.shape[1]) # zeros std 
            cell_statistics_class_wise.extend([None] * property_list.shape[1]) # zeros skew 
            cell_statistics_class_wise.extend([None] * property_list.shape[1]) # zeros kurtosis 
            continue

        # prepare 
        per_class_properties = points_properties[np.where(points_type==i)[0]].copy()
        per_class_properties = np.sort(per_class_properties, axis=0) # sort by value 
        n = int(outlier_removal * len(per_class_properties)) 
        per_class_properties = per_class_properties[n : len(per_class_properties)-n] # remove lower/upper outliers 

        # calculate statistics for each class type 
        cell_statistics_class_wise.extend(per_class_properties.mean(0))
        cell_statistics_class_wise.extend(per_class_properties.std(0))
        cell_statistics_class_wise.extend(skew(cell_statistics_class_wise, axis=0)) # skew
        cell_statistics_class_wise.extend(kurtosis(cell_statistics_class_wise, axis=0)) # kurtosis 
    
    cell_statistics_class_wise = np.array(cell_statistics_class_wise)
    return cell_statistics_class_wise


### 

# ... (run_extraction, prepare_and_save, main)
# 생략

