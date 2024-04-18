import nibabel as nib
import numpy as np
import os
import math
# Load the input NIfTI file
nii_file_path = './detetection/dataset/for_nifti_sengmentation/nifti_file/test/00428/FLAIR.nii'
nii_file_path_output = './detetection/sengmentation'

def convert_nifti_volume(nifti_file_path, output_path):
    """
    This function takes a path to a NIfTI file, loads the data, and pads the data array
    with zeroes to ensure that the 3 axes (i.e. dimensions) are all 155 in length.

    The padding is done equally on both sides of the data array.

    Parameters
    ----------
    nifti_file_path : str
        The path to the input NIfTI file.
    output_path : str
        The directory path to save the modified NIfTI file.

    Returns
    -------
    str
        The path to the modified NIfTI file.
    """
    nifti = nib.load(nifti_file_path)
    data = nifti.get_fdata()
    axis_0_length = data.shape[0]
    axis_1_length = data.shape[1]
    axis_2_length = data.shape[2]

    # Check which axis needs to be padded and pad it equally on both sides
    if axis_0_length < 155:  # Pad axis 0
        num_slices_before = math.ceil((155 - axis_0_length) / 2)
        num_slices_after = math.floor((155 - axis_0_length) / 2)
        pad_before = np.zeros((num_slices_before, axis_1_length, axis_2_length))
        pad_after = np.zeros((num_slices_after, axis_1_length, axis_2_length))
        padded_data = np.concatenate((pad_before, data, pad_after), axis=0)
    elif axis_1_length < 155:  # Pad axis 1
        num_slices_before = math.ceil((155 - axis_1_length) / 2)
        num_slices_after = math.floor((155 - axis_1_length) / 2)
        pad_before = np.zeros((axis_0_length, num_slices_before, axis_2_length))
        pad_after = np.zeros((axis_0_length, num_slices_after, axis_2_length))
        padded_data = np.concatenate((pad_before, data, pad_after), axis=1)
    elif axis_2_length < 155:  # Pad axis 2
        num_slices_before = math.ceil((155 - axis_2_length) / 2)
        num_slices_after = math.floor((155 - axis_2_length) / 2)
        pad_before = np.zeros((axis_0_length, axis_1_length, num_slices_before))
        pad_after = np.zeros((axis_0_length, axis_1_length, num_slices_after))
        padded_data = np.concatenate((pad_before, data, pad_after), axis=2)
    else:
        padded_data = data

    # Create a new NIfTI file with the modified header and data
    header = nifti.header.copy()
    header.set_data_shape(padded_data.shape)
    modified_nifti = nib.Nifti1Image(padded_data, nifti.affine, header=header)
    modified_file_path = os.path.join(output_path, 'modified_file.nii')
    nib.save(modified_nifti, modified_file_path)
    return modified_file_path


def main():
    convert_nifti_volume(nii_file_path, nii_file_path_output)