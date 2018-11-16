"""
Data Reading And Manipulation Utilities
"""

import os
import numpy as np
import pandas as pd


from skimage.io import imread

IMG_SCALING = (1, 1)

############################################################
#  Bounding Boxes Encode To Image and Decode To RLE 
############################################################

def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

############################################################
#  Read Dataset
############################################################


def getImageIdAndShipMasks(segmentationFile, train_image_directory):
    shipmasks = pd.read_csv(segmentationFile)
    shipmasks = shipmasks[shipmasks['EncodedPixels'].notnull()]
    shipmasks['ships'] = shipmasks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    unique_ids = shipmasks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    shipmasks.drop('ships', 1,inplace = True)
    unique_ids['has_ship'] = unique_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
    # some files are too small/corrupt
    unique_ids['file_size_kb'] = unique_ids['ImageId'].map(lambda c_img_id: 
                                                               os.stat(os.path.join(train_image_directory, 
                                                                                    c_img_id)).st_size/1024)
    unique_ids = unique_ids[unique_ids['file_size_kb']>60] # keep only 50kb files
    return unique_ids, shipmasks

############################################################
#  Random Image Generator
############################################################


def make_image_gen( train_image_directory, mask_df, sampling_df, batch_size = 1):
    out_rgb = []
    out_mask = []
    while True: 
        single_batch =sampling_df.sample(batch_size)
        imgs =single_batch['ImageId'].values
        for idx in range(batch_size):
            rgb_path = os.path.join(train_image_directory, imgs[idx])
            c_img = imread(rgb_path)
            msks =mask_df[mask_df['ImageId']==imgs[idx]]['EncodedPixels'].values
            c_mask = masks_as_image(msks)
            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
            out_rgb += [c_img]
            out_mask += [c_mask] 
        yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
        out_rgb, out_mask=[], []
        

        
def create_aug_gen(in_gen, i_gen, l_gen, seed = None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    #gc.collect()
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = i_gen.flow(255*in_x, 
                             batch_size = in_x.shape[0], 
                             seed = seed, 
                             shuffle=True)
        g_y = l_gen.flow(in_y, 
                             batch_size = in_x.shape[0], 
                             seed = seed, 
                             shuffle=True)

        yield next(g_x)/255.0, next(g_y)
