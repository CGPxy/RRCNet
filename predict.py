import cv2
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os
import argparse
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder  
from model_net import *
mpl.use('TkAgg')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TEST_SET = []

def read_directory(directory_name):
    for filename in os.listdir(directory_name):
        TEST_SET.append(filename)

def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="path to trained model model")
    args = vars(ap.parse_args())    
    return args


def original_predict(args):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = RRCNet()
    # model = load_model("/home/dy/CGP/Unet_Segnet/compare_mothds/unet1/128_model/" + args["model"])
    model.summary()
    model.load_weights("/media/dy/Data_2T/CGP/Unet_Segnet/method/BASNet/Ours/model/DatasetB/4/" + args["model"], by_name=True)

    start_time = time.clock()
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
       
        # BUSIS
        image = cv2.imread('/media/dy/Data_2T/CGP/Unet_Segnet/data/new_Dataset_B/4/Test_images/images/384/' + path)

        #DatasetB
        # image = cv2.imread('/media/dy/Data_2T/CGP/Unet_Segnet/data/new_Dataset_B/384/benign/images/' + path)

        print(image.shape)
        image = np.array(image,dtype=np.uint8)
        h,w,_ = image.shape
        
        padding_h = h
        padding_w = w
        padding_img = np.zeros((padding_h, padding_w, 3),dtype=np.uint8)
        padding_img[0:h,0:w,:] = image[:,:,:]
        padding_img = padding_img.astype("float") / 255.0
        padding_img = img_to_array(padding_img)

        mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
   
        crop = padding_img[:,:,:3]
            
        crop = np.expand_dims(crop, axis=0) 
        pred0, pred1, pred2,pred3,pred4,pred5,pred6,pred7 = model.predict(crop,verbose=2)

        preimage = pred.reshape((384,384)) # * 255

        h,w = preimage.shape
        for i in range(0, h):
            for j in range(0, w):
                if (preimage[i, j] > 0.5):
                    preimage[i, j] = 1
                else:
                    preimage[i, j] = 0

        pred = preimage.reshape((384,384)).astype(np.uint8)

        mask_whole[:,:] = pred[:,:]

        BUSIS
        cv2.imwrite('/media/dy/Data_2T/CGP/Unet_Segnet/method/BASNet/Ours/result/mask/Ablation/Segnet_DS_MR/4/'+path,mask_whole[0:h,0:w])

        # DatasetB
        # cv2.imwrite('/media/dy/Data_2T/CGP/Unet_Segnet/method/BASNet/Ours/result/mask/benign/DatasetB/1/'+path,mask_whole[0:h,0:w])
    print(time.clock()-start_time)

if __name__ == '__main__':
    # BUSIS
    read_directory("/media/dy/Data_2T/CGP/Unet_Segnet/data/new_Dataset_B/4/Test_images/images/384/")

    # DatasetB
    # read_directory("/media/dy/Data_2T/CGP/Unet_Segnet/data/new_Dataset_B/384/benign/images/")
    args = args_parse()
    original_predict(args)



