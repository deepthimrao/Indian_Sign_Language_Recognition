''' Given a dictionary (key-label, value-list of images for the label), 
converts it into the format needed for training'''

import numpy as np
import os
import pandas as pd
import cv2
import segmentation
import sys
import random
from skimage.feature import hog


def vectorize_image(img_arr):
    r,c = img_arr.shape
    return np.reshape(img_arr,(r*c),order='C')

def create_data(data):
    dataset_labels = []
    dataset_images = []
    df = pd.DataFrame()
    for label in data:
        # print(label)
        images_list = data[label]
        for img_arr in images_list:
            
            vec_img = vectorize_image(img_arr)
            dataset_labels.append(label)
            dataset_images.append(list(vec_img))
    
    print(len(dataset_images))
    df = df.append(dataset_images)
    del dataset_images
    return df,dataset_labels
    
    # return dataset_labels

    
def mat2gray(img):
    A = np.double(img)
    out = np.zeros(A.shape, np.double)
    normalized = cv2.normalize(A, out, 1.0, 0.0, cv2.NORM_MINMAX)
    return out


def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
    image = mat2gray(image)
    
    mode = mode.lower()
    if image.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
    if seed is not None:
        np.random.seed(seed=seed)
        
    if mode == 'gaussian':
        noise = np.random.normal(kwargs['mean'], kwargs['var'] ** 0.5,
                                 image.shape)        
        out = image  + noise
    if clip:        
        out = np.clip(out, low_clip, 1.0)
        
    return out

def add_noise(img):
  
    # Getting the dimensions of the image
    row , col = img.shape
      
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(600, 650)
    for i in range(number_of_pixels):
        
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
          
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
          
        # Color that pixel to white
        img[y_coord][x_coord] = 255
          
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    # number_of_pixels = random.randint(100 , 300)
    # for i in range(number_of_pixels):
        
    #     # Pick a random y coordinate
    #     y_coord=random.randint(0, row - 1)
          
    #     # Pick a random x coordinate
    #     x_coord=random.randint(0, col - 1)
          
    #     # Color that pixel to black
    #     img[y_coord][x_coord] = 0
          
    return img

def get_background_images(size):
    bg_images = []
    for img in os.listdir('bg_change/'):
        bg = cv2.imread('bg_change/'+img)
        bg = cv2.resize(bg,size)
        bg_images.append(bg)
        
    bg = np.random.randint(0,255,size=(size[0],size[1],3))
    bg = np.array(bg,dtype=np.uint8)
    bg_images.append(bg)
    
    return bg_images

def generate_random_bg(bg_images):
    
    return bg_images[np.random.randint(0,len(bg_images))]
    
       
def add_random_bg(image,bg):
    
    out1 = segmentation.segment(image)
    out1 = cv2.bitwise_and(image,image,mask=out1)
    
    out2 = segmentation.segment(image)
    out2 = cv2.bitwise_not(out2)
    out2 = cv2.bitwise_and(bg,bg,mask=out2)
    
    out = out1+out2
    out_gray = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
    
    return out_gray

CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

dataset_path = 'dataset/'
data = {}

bg_images = get_background_images((64,64))

for label in CLASSES:
    images_list = []
    for img in os.listdir(dataset_path+label):
        # print(img)
        
        image = cv2.imread(dataset_path+label+'/'+img)
        image = cv2.resize(image,(64,64))
        
        bg = generate_random_bg(bg_images)
        output_image = add_random_bg(image,bg)
        # fd, binary_hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
        #             cells_per_block=(1, 1), visualize=True)
        
        # binary_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # binary_image = segmentation.segment(image)
        # binary_image = cv2.bitwise_and(image,image,mask=binary_image)
        # binary_image = cv2.cvtColor(binary_image,cv2.COLOR_BGR2GRAY)
        # binary_image = random_noise(binary_image,mean=0.1,var=0.001)
        # binary_image = segmentation.segment(binary_image)
        # binary_image = add_noise(binary_image)
        
        images_list.append(output_image)
        
        # binary_images_list.append(binary_hog_image)
        
        # cv2.namedWindow("test",cv2.WINDOW_NORMAL)
        # cv2.imshow("test",output_image)
        # print(output_image)
        # if(cv2.waitKey(0) == ord('a')):
        #     cv2.destroyAllWindows()
        #     sys.exit()
        # break
    
    data[label] = images_list
    
cv2.destroyAllWindows()    
df,df_labels = create_data(data)
df.to_csv('isl_data_rand_bg_2.csv',index=False)
a = pd.DataFrame(df_labels)
a.to_csv("labels.csv",index=False)
