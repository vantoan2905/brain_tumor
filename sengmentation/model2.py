


# In[2]:


# Nhập các thư viện cần thiết

import os
import cv2
import glob
import PIL
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage 
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from PIL import Image, ImageOps  
import nilearn as nl 
import nibabel as nib 
import nilearn.plotting as nlplt 
# !pip install https://github.com/miykael/gif_your_nifti.git # nifti to gif 

import gif_your_nifti.core as gif2nif 


import keras
import keras.backend as K
from keras.layers import Input, Flatten
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras import preprocessing


np.set_printoptions(precision=3, suppress=True)



# In[3]:


SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', 
    2 : 'EDEMA',
    3 : 'ENHANCING' 
}

VOLUME_SLICES = 100 
VOLUME_START_AT = 22 


# In[4]:


import os
TRAIN_DATASET_PATH = os.path.join('E:\\learn\\nkkh\\detetection\\dataset\\for_nifti_sengmentation\\data_brast_2020\\data\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData/')
VALIDATION_DATASET_PATH = os.path.join('E:\\learn\\nkkh\\detetection\\dataset\\for_nifti_sengmentation\\data_brast_2020\\data\\BraTS2020_ValidationData\\MICCAI_BraTS2020_ValidationData/')

# test_image_flair=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii').get_fdata()
# test_image_t1=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1.nii').get_fdata()
# test_image_t1ce=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii').get_fdata()
# test_image_t2=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t2.nii').get_fdata()
# test_mask=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii').get_fdata()

# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (20, 10))
# slice_w = 25
# ax1.imshow(test_image_flair[:,:,test_image_flair.shape[0]//2-slice_w], cmap = 'gray')
# ax1.set_title('Image flair')
# ax2.imshow(test_image_t1[:,:,test_image_t1.shape[0]//2-slice_w], cmap = 'gray')
# ax2.set_title('Image t1')
# ax3.imshow(test_image_t1ce[:,:,test_image_t1ce.shape[0]//2-slice_w], cmap = 'gray')
# ax3.set_title('Image t1ce')
# ax4.imshow(test_image_t2[:,:,test_image_t2.shape[0]//2-slice_w], cmap = 'gray')
# ax4.set_title('Image t2')
# ax5.imshow(test_mask[:,:,test_mask.shape[0]//2-slice_w])
# ax5.set_title('Mask')



# # In[5]:

# fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
# ax1.imshow(rotate(montage(test_image_t1[50:-50,:,:]), 90, resize=True), cmap ='gray')

# # In[6]:

# fig, ax1 = plt.subplots(1, 1, figsize = (15,15))

# ax1.imshow(rotate(montage(test_mask[60:-60,:,:]), 90, resize=True), cmap ='gray')

# # In[7]:

# shutil.copy2(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii', './test_gif_BraTS20_Training_001_flair.nii')
# gif2nif.write_gif_normal('./test_gif_BraTS20_Training_001_flair.nii')

# # In[8]:


# niimg = nl.image.load_img(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii')
# nimask = nl.image.load_img(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii')

# fig, axes = plt.subplots(nrows=4, figsize=(30, 40))


# nlplt.plot_anat(niimg,
#                 title='BraTS20_Training_001_flair.nii plot_anat',
#                 axes=axes[0])

# nlplt.plot_epi(niimg,
#                title='BraTS20_Training_001_flair.nii plot_epi',
#                axes=axes[1])

# nlplt.plot_img(niimg,
#                title='BraTS20_Training_001_flair.nii plot_img',
#                axes=axes[2])

# nlplt.plot_roi(nimask, 
#                title='BraTS20_Training_001_flair.nii with mask plot_roi',
#                bg_img=niimg, 
#                axes=axes[3], cmap='Paired')

# plt.show()

# In[9]:


def dice_coef(y_true, y_pred, smooth=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.

    Parameters
    ----------
    y_true: array-like
        Ground truth labels.
    y_pred: array-like
        Predicted labels.
    smooth: float
        Smoothing constant for the dice coefficient.

    Returns
    -------
    float
        Dice coefficient.

    """
    class_num = 4

    # Compute per-class intersection and union areas.
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)

        # Compute the dice coefficient for this class.
        loss = ((2. * intersection + smooth) /
                (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

        # Print dice coefficient for each class.
        K.print_tensor(loss, message='loss value for class {} : '.format(SEGMENT_CLASSES[i]))

        # Add to the total loss.
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss

    # Average the loss over classes.
    total_loss = total_loss / class_num

    return total_loss


 
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    """Compute the dice coefficient for the necrotic class.

    The dice coefficient is a measure of the similarity between two label
    distributions. It ranges from 0 (no similarity) to 1 (perfect match).

    Arguments:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        epsilon: Smoothing constant to avoid division by zero.

    Returns:
        Dice coefficient.
    """
    intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)
    # return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    """Compute the dice coefficient for the edema class.

    The dice coefficient is a measure of the similarity between two label
    distributions. It ranges from 0 (no similarity) to 1 (perfect match).

    Arguments:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        epsilon: Smoothing constant to avoid division by zero.

    Returns:
        Dice coefficient.
    """

    # Intersection is the number of pixels that are the same in both y_true and
    # y_pred. Convert to float() so that division is floating-point division.
    intersection = K.sum(K.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))

    # Denominator is the total number of pixels in y_true plus y_pred. The
    # epsilon value is added to avoid division by zero.
    denominator = K.sum(K.square(y_true[:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,2])) + epsilon

    # Dice coefficient is (2 * intersection) / (denominator).
    return (2. * intersection) / denominator

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    """Compute the dice coefficient for the enhancing tumor class.

    The dice coefficient is a measure of the similarity between two label
    distributions. It ranges from 0 (no similarity) to 1 (perfect match).

    Arguments:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        epsilon: Smoothing constant to avoid division by zero.

    Returns:
        Dice coefficient.
    """

    # Intersection is the number of pixels that are the same in both y_true and
    # y_pred. Convert to float() so that division is floating-point division.
    intersection = K.sum(K.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))

    # Denominator is the total number of pixels in y_true plus y_pred. The
    # epsilon value is added to avoid division by zero.
    denominator = K.sum(K.square(y_true[:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,3])) + epsilon

    # Dice coefficient is (2 * intersection) / (denominator).
    return (2. * intersection) / denominator



 
def precision(y_true, y_pred):
    """Compute precision for the given true and predicted labels.

    Precision is the ratio of True Positives to True Positives plus False Positives.
    It measures the ratio of correctly identified positive samples out of all
    positive samples that were identified as positive.

    Arguments:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Precision metric.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

    
# Tính toán độ nhạy      
def sensitivity(y_true, y_pred):
    """
    Compute sensitivity for the given true and predicted labels.

    Sensitivity is the ratio of True Positives to True Positives plus False Negatives.
    It measures the ratio of correctly identified positive samples out of all
    actual positive samples.

    Arguments:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Sensitivity metric.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


#Tính toán tính đặc hiệu
def specificity(y_true, y_pred):
    """Compute specificity for the given true and predicted labels.

    Specificity is the ratio of True Negatives to True Negatives plus False Positives.
    It measures the ratio of correctly identified negative samples out of all
    actual negative samples.

    Arguments:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Specificity metric.
    """
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


# In[10]:


IMG_SIZE=128


# In[11]:


def build_unet(inputs, ker_init, dropout):
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv1)
    
    pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool)
    conv = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv3)
    
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv5)
    drop5 = Dropout(dropout)(conv5)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(drop5))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv9)
    
    up = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv9))
    merge = concatenate([conv1,up], axis = 3)
    conv = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge)
    conv = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv)
    
    conv10 = Conv2D(4, (1,1), activation = 'softmax')(conv)
    
    return Model(inputs = inputs, outputs = conv10)

input_layer = Input((IMG_SIZE, IMG_SIZE, 2))

model = build_unet(input_layer, 'he_normal', 0.2)
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing] )


# In[12]:


train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]

train_and_val_directories.remove(TRAIN_DATASET_PATH+'BraTS20_Training_355')


def pathListIntoIds(dirList):
    """
    Takes a list of directory paths and returns a list of the directory names

    Parameters
    ----------
    dirList : list
        List of directory paths

    Returns
    -------
    list
        List of directory names
    """
    x = []
    for i in range(0,len(dirList)):
        # For each directory path, find the last '/' and return the substring
        # from there to the end of the string
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x



train_and_test_ids = pathListIntoIds(train_and_val_directories); 

    
train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.2) 
train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15) 



# In[13]:


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, dim=(IMG_SIZE,IMG_SIZE), batch_size = 1, n_channels = 2, shuffle=True):
        """
        Class for generating training data in batches

        Parameters
        ----------
        list_IDs : list
            List of case identifiers
        dim : tuple, optional
            Dimensions of input and output data, by default (IMG_SIZE,IMG_SIZE)
        batch_size : int, optional
            Batch size, by default 1
        n_channels : int, optional
            Number of input channels, by default 2
        shuffle : bool, optional
            Whether to shuffle the data at the end of each epoch, by default True
        """
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Calculates the number of batches in the dataset.
        The number of batches is calculated as the floor of the
        division of the number of cases by the batch size.

        Returns
        -------
        int
            Number of batches in the dataset
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generates a batch of data.

        Parameters
        ----------
        index : int
            Index of the batch

        Returns
        -------
        X : numpy.ndarray
            Input data for the network (size: batch_size, image_height, image_width, n_channels)
        y : numpy.ndarray
            Output data for the network (size: batch_size, image_height, image_width)
        """
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        Batch_ids = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        X = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size*VOLUME_SLICES, 240, 240))
        Y = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, 4))

        
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(TRAIN_DATASET_PATH, i)

            data_path = os.path.join(case_path, f'{i}_flair.nii');
            flair = nib.load(data_path).get_fdata()    

            data_path = os.path.join(case_path, f'{i}_t1ce.nii');
            ce = nib.load(data_path).get_fdata()
            
            data_path = os.path.join(case_path, f'{i}_seg.nii');
            seg = nib.load(data_path).get_fdata()
        
            for j in range(VOLUME_SLICES):
                 X[j +VOLUME_SLICES*c,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));
                 X[j +VOLUME_SLICES*c,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));

                 y[j +VOLUME_SLICES*c] = seg[:,:,j+VOLUME_START_AT];
                    
        # Generate masks
        y[y==4] = 3;
        mask = tf.one_hot(y, 4);
        Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE));
        return X/np.max(X), Y
        
training_generator = DataGenerator(train_ids)
valid_generator = DataGenerator(val_ids)
test_generator = DataGenerator(test_ids)



# In[14]:


# show number of data for each dir 
def showDataLayout():
    plt.bar(["Train","Valid","Test"],
    [len(train_ids), len(val_ids), len(test_ids)], align='center',color=[ 'green','red', 'blue'])
    plt.legend()

    plt.ylabel('Number of images')
    plt.title('Data distribution')

    plt.show()
    
showDataLayout()


# In[15]:


csv_logger = CSVLogger('training.log', separator=',', append=False)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
                              patience=2, verbose=1, mode='auto'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.000001, verbose=1),
    # keras.callbacks.ModelCheckpoint(filepath = 'model_.{epoch:02d}-{val_loss:.6f}.weights.h5',
    #                         verbose=1, save_best_only=True, save_weights_only = True),
    csv_logger
]


# In[16]:


K.clear_session()

history =  model.fit(training_generator,
                    epochs=1,
                    steps_per_epoch=len(train_ids),
                    callbacks= callbacks,
                    validation_data = valid_generator
                    )  
model.save("model_x1_1.h5")





# In[1]:


# Load the model
model = keras.models.load_model(
    r'E:\\learn\\nkkh\\detetection\\sengmentation\\model_x1_1.h5',
    custom_objects={
        'accuracy': tf.keras.metrics.MeanIoU(num_classes=4),
        'dice_coef': dice_coef,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'dice_coef_necrotic': dice_coef_necrotic,
        'dice_coef_edema': dice_coef_edema,
        'dice_coef_enhancing': dice_coef_enhancing
    },
    compile=False
)

# Load the training history
hist = pd.read_csv(
    r'E:\\learn\\nkkh\\detetection\\sengmentation\\training.log',
    sep=',',
    engine='python'
)

epoch = hist['epoch']
acc = hist['accuracy']
val_acc = hist['val_accuracy']
loss = hist['loss']
val_loss = hist['val_loss']
train_dice = hist['dice_coef']
val_dice = hist['val_dice_coef']

# Plotting
f, ax = plt.subplots(1, 4, figsize=(16, 8))

ax[0].plot(epoch, acc, 'b', label='Training Accuracy')
ax[0].plot(epoch, val_acc, 'r', label='Validation Accuracy')
ax[0].legend()

ax[1].plot(epoch, loss, 'b', label='Training Loss')
ax[1].plot(epoch, val_loss, 'r', label='Validation Loss')
ax[1].legend()

ax[2].plot(epoch, train_dice, 'b', label='Training dice coef')
ax[2].plot(epoch, val_dice, 'r', label='Validation dice coef')
ax[2].legend()

# Corrected column names for mean IOU
ax[3].plot(epoch, hist['mean_iou'], 'b', label='Training mean IOU')
ax[3].plot(epoch, hist['val_mean_iou'], 'r', label='Validation mean IOU')
ax[3].legend()

plt.show()


# In[18]:



def imageLoader(path):
    """
    Load nifti file at `path`
    and return a numpy array of shape
    (batch_size * n_slices, x, y, n_channels)
    where n_channels is the number of channels in the input data
    and each slice is resized to (IMG_SIZE, IMG_SIZE)
    """
    image = nib.load(path).get_fdata()
    X = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, self.n_channels))
    for j in range(VOLUME_SLICES):
        # resize input data to (IMG_SIZE, IMG_SIZE)
        X[j +VOLUME_SLICES*c,:,:,0] = cv2.resize(image[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        # resize corrupted data to (IMG_SIZE, IMG_SIZE)
        X[j +VOLUME_SLICES*c,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        # each slice is a binary mask
        # so y is a numpy array of shape (batch_size * n_slices, x, y)
        y[j +VOLUME_SLICES*c] = seg[:,:,j+VOLUME_START_AT]
    return np.array(image)


def loadDataFromDir(path, list_of_files, mriType, n_images):
    """
    Load nifti files from path with names in list_of_files[:n_images]
    and return pair of numpy arrays:
    (X, y) where X is a numpy array of shape
    (n_images * n_slices, x, y, n_channels)
    and y is a numpy array of shape
    (n_images * n_slices, x, y)
    where n_slices is the number of slices in each input nifti file,
    x and y are image dimensions after resizing
    and n_channels is the number of channels in the input data.

    Arguments:
        path: path to directory with nifti files
        list_of_files: list of file names in directory
        mriType: string which will be used to filter
            nifti files by their names
        n_images: number of nifti files to load
    """
    scans = []
    masks = []
    for i in list_of_files[:n_images]:
        fullPath = glob.glob(os.path.join(path, i + '/*' + mriType + '*'))[0]
        currentScanVolume = imageLoader(fullPath)
        currentMaskVolume = imageLoader(glob.glob(os.path.join(path, i + '/*seg*'))[0])
        # for each slice in 3D volume, find also it's mask
        for j in range(0, currentScanVolume.shape[2]):
            scan_img = cv2.resize(currentScanVolume[:,:,j], dsize=(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA).astype('uint8')
            mask_img = cv2.resize(currentMaskVolume[:,:,j], dsize=(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA).astype('uint8')
            scans.append(scan_img[..., np.newaxis])
            masks.append(mask_img[..., np.newaxis])
    return np.array(scans, dtype='float32'), np.array(masks, dtype='float32')

        
#brains_list_test, masks_list_test = loadDataFromDir(VALIDATION_DATASET_PATH, test_directories, "flair", 5)




# In[22]:


def predictByPath(case_path, case):
    """
    Predict the mask of a given case_path

    Parameters
    ----------
    case_path : str
        The path of the case to predict
    case : str
        The name of the case

    Returns
    -------
    preds : numpy.ndarray
        The predicted mask of the case
    """
    # path of the case
    files = next(os.walk(case_path))[2]
    # initialize the inputs volume with the number of slices
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
    # initialize the outputs volume with the number of slices
  #  y = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE))
    
    # path of the flair image
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_flair.nii')
    # load the flair image
    flair=nib.load(vol_path).get_fdata()
    
    # path of the ce image
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_t1ce.nii')
    # load the ce image
    ce=nib.load(vol_path).get_fdata() 
    
    # path of the segmentation
 #   vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_seg.nii');
    # load the segmentation
 #   seg=nib.load(vol_path).get_fdata()  

    
    # iterate over the slices in the volume
    for j in range(VOLUME_SLICES):
        # resize the flair slice and append it to the inputs volume
        X[j,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))
        # resize the ce slice and append it to the inputs volume
        X[j,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))
        # resize the segmentation slice and append it to the outputs volume
 #       y[j,:,:] = cv2.resize(seg[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))
        
  #  model.evaluate(x=X,y=y[:,:,:,0], callbacks= callbacks)
    # predict the mask of the case
    return model.predict(X/np.max(X), verbose=1)



def showPredictsById(case, start_slice=60):
    """Shows predictions for a given case (id)
    
    This function loads data, make predictions and show them
    
    Args:
        case (str): id of the case to be shown
        start_slice (int): the slice to start showing from. Defaults to 60.
    """
    path = f"E:\\learn\\nkkh\\detetection\\dataset\\for_nifti_sengmentation\\data_brast_2020\\data\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_{case}"
    gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
    origImage = nib.load(os.path.join(path, f'BraTS20_Training_{case}_flair.nii')).get_fdata()
    p = predictByPath(path,case)

    core = p[:,:,:,1]
    edema= p[:,:,:,2]
    enhancing = p[:,:,:,3]

    plt.figure(figsize=(18, 50))
    f, axarr = plt.subplots(1,6, figsize = (18, 50)) 

    for i in range(6): # for each image, add brain background
        axarr[i].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray", interpolation='none')
        
    axarr[0].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
    axarr[0].set_title('Original image flair')
    curr_gt=cv2.resize(gt[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_NEAREST)
    axarr[1].imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3) # ,alpha=0.3,cmap='Reds'
    axarr[1].set_title('Ground truth')
    axarr[2].imshow(p[start_slice,:,:,1:4], cmap="Reds", interpolation='none', alpha=0.3)
    axarr[2].set_title('all classes')
    axarr[3].imshow(edema[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[3].set_title(f'{SEGMENT_CLASSES[1]} predicted')
    axarr[4].imshow(core[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[4].set_title(f'{SEGMENT_CLASSES[2]} predicted')
    axarr[5].imshow(enhancing[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[5].set_title(f'{SEGMENT_CLASSES[3]} predicted')
    plt.show()
    
    
showPredictsById(case=test_ids[0][-3:])
showPredictsById(case=test_ids[1][-3:])
showPredictsById(case=test_ids[2][-3:])
showPredictsById(case=test_ids[3][-3:])
showPredictsById(case=test_ids[4][-3:])
showPredictsById(case=test_ids[5][-3:])
showPredictsById(case=test_ids[6][-3:])





# In[24]:


case = case=test_ids[3][-3:]
path = f"E:\\learn\\nkkh\\detetection\\dataset\\for_nifti_sengmentation\\data_brast_2020\\data\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_{case}"
gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
p = predictByPath(path,case)


core = p[:,:,:,1]
edema= p[:,:,:,2]
enhancing = p[:,:,:,3]


i=40 # slice at
eval_class = 2 #     0 : 'NOT tumor',  1 : 'ENHANCING',    2 : 'CORE',    3 : 'WHOLE'



gt[gt != eval_class] = 1 # use only one class for per class evaluation 

resized_gt = cv2.resize(gt[:,:,i+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))

plt.figure()
f, axarr = plt.subplots(1,2) 
axarr[0].imshow(resized_gt, cmap="gray")
axarr[0].title.set_text('ground truth')
axarr[1].imshow(p[i,:,:,eval_class], cmap="gray")
axarr[1].title.set_text(f'predicted class: {SEGMENT_CLASSES[eval_class]}')
plt.show()


# In[25]:


model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing] )
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_generator, batch_size=100, callbacks= callbacks)
print("test loss, test acc:", results)



