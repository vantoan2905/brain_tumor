import dicom2nifti

def dicom_to_nifti(dicom_dir, output_dir):
    """
    Chuyển đổi các file DICOM trong thư mục 'dicom_dir' thành định dạng NIfTI
    và lưu chúng vào thư mục 'output_dir'.
    """
    dicom2nifti.convert_directory(dicom_dir, output_dir)

# Sử dụng hàm
dicom_directory = "/path/to/dicom_directory"
output_directory = "/path/to/output_directory"
dicom_to_nifti(dicom_directory, output_directory)
