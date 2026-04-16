import pickle
import numpy as np
# use skimgage/scipy to extract basic cell level features (shape, intensity, texture)
from skimage.measure import regionprops, label, regionprops_table
from skimage.color import rgb2gray 
from scipy.stats import skew, kurtosis 
from skimage.feature import graycomatrix, graycoprops
"""
각 cell 마다: 
    - Shape: area, perimeter, orientation, eccentricity, solidity, axis_major_length, axis_minor_length
    - Intensity: mean, std, skew, kurtosis 
    - Texture: GLCM - contrast, dissimilarity, homogeneity, energy 
"""

### 

def extract_single_cell_properties(
    cell_mask: np.ndarray,
    cell_rgb_crop: np.ndarray,
) -> list[float]:
    """
    Extract handcrafted features for a single cell.

    Args:
        cell_mask: binary mask for one cell, shape (H, W), values {0,1} or bool
        cell_rgb_crop: RGB image crop containing the cell, shape (H, W, 3)

    Returns:
        props_list: list of handcrafted features
            [shape features..., intensity features..., texture features...]
    """
    props_list: list[float] = []

    # ensure mask is binary uint8
    cell_mask = (cell_mask > 0).astype(np.uint8)

    # -------------------------
    # (1) Shape features
    # -------------------------
    shape_properties = [
        "area",
        "perimeter",
        "orientation",
        "eccentricity",
        "solidity",
        "axis_major_length",
        "axis_minor_length",
    ]

    shape_dict = regionprops_table(
        label_image=cell_mask,
        properties=shape_properties,
    )

    # regionprops_table returns arrays, even for one object
    # extract scalar values in the given order
    for prop_name in shape_properties:
        values = shape_dict.get(prop_name, [np.nan])
        props_list.append(float(values[0]) if len(values) > 0 else np.nan)

    # -------------------------
    # (2) Intensity features
    # -------------------------
    if cell_rgb_crop.ndim == 3:
        gray_crop = rgb2gray(cell_rgb_crop)  # float in [0,1]
        gray_crop = (gray_crop * 255).astype(np.uint8)
    else:
        # already grayscale
        gray_crop = cell_rgb_crop.astype(np.uint8)

    masked_pixels = gray_crop[cell_mask == 1]

    if masked_pixels.size == 0:
        props_list.extend([np.nan, np.nan, np.nan, np.nan])
    else:
        props_list.append(float(masked_pixels.mean()))
        props_list.append(float(masked_pixels.std()))
        props_list.append(float(skew(masked_pixels)) if masked_pixels.size > 2 else np.nan)
        props_list.append(float(kurtosis(masked_pixels)) if masked_pixels.size > 3 else np.nan)

    # -------------------------
    # (3) Texture features
    # -------------------------
    # GLCM expects a 2D grayscale image.
    # To focus on the cell, keep masked region and zero out background.
    gray_for_glcm = gray_crop.copy()
    gray_for_glcm[cell_mask == 0] = 0

    glcm = graycomatrix(
        gray_for_glcm,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )

    props_list.extend([
        float(graycoprops(glcm, "contrast")[0, 0]),
        float(graycoprops(glcm, "dissimilarity")[0, 0]),
        float(graycoprops(glcm, "homogeneity")[0, 0]),
        float(graycoprops(glcm, "energy")[0, 0]),
    ])

    return props_list


def build_cell_property_dict(
    pred_data: dict,
    cell_mask_dict: dict,
    cell_crop_dict: dict,
    mag_ratio: float,
    type_key: str = "type",
) -> dict:
    """
    Build dictionary of cell-level handcrafted features.

    Args:
        pred_data:
            expected structure:
            pred_data['nuc'][cell_id]['centroid']
            pred_data['nuc'][cell_id][type_key]
        cell_mask_dict:
            {cell_id: binary_mask_for_that_cell}
        cell_crop_dict:
            {cell_id: RGB crop for that cell}
        mag_ratio:
            used to map centroid coordinates to desired resolution
        type_key:
            key name for nucleus type in pred_data['nuc'][cell_id]

    Returns:
        prop_dict:
            {
              cell_id: {
                "centroid": ...,
                "type": ...,
                "properties": [...]
              },
              ...
            }
    """
    prop_dict = {}

    for cell_id in pred_data["nuc"].keys():
        if cell_id not in cell_mask_dict:
            continue
        if cell_id not in cell_crop_dict:
            continue

        cell_mask = cell_mask_dict[cell_id]
        cell_rgb_crop = cell_crop_dict[cell_id]

        props_list = extract_single_cell_properties(
            cell_mask=cell_mask,
            cell_rgb_crop=cell_rgb_crop,
        )

        prop_dict[cell_id] = {}
        prop_dict[cell_id]["centroid"] = (
            np.array(pred_data["nuc"][cell_id]["centroid"]) // mag_ratio
        )
        prop_dict[cell_id]["type"] = pred_data["nuc"][cell_id][type_key]
        prop_dict[cell_id]["properties"] = props_list

    return prop_dict


def save_cell_property_dict(
    prop_dict: dict,
    save_path: str,
) -> None:
    with open(save_path, "wb") as f:
        pickle.dump(prop_dict, f)


# --------------------------------------------------
# Example usage
# --------------------------------------------------
# pred_data['nuc'][cell_id]['centroid']
# pred_data['nuc'][cell_id]['type']
#
# cell_mask_dict[cell_id] = single-cell binary mask
# cell_crop_dict[cell_id] = corresponding RGB crop
#
# prop_dict = build_cell_property_dict(
#     pred_data=pred_data,
#     cell_mask_dict=cell_mask_dict,
#     cell_crop_dict=cell_crop_dict,
#     mag_ratio=mag_ratio,
#     type_key="type",
# )
#
# save_cell_property_dict(prop_dict, "/path/to/save/pickle/file.pkl")
#
# del prop_dict
# del pred_data
