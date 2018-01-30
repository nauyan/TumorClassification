from glob import glob
from skimage import io
import numpy as np
np.set_printoptions(threshold=np.nan)
import cv2
import progressbar
import os

t1 = glob('../HGG/brats*/*T1.*/*.mha')
directory='T1Images'
if not os.path.exists(directory):
    os.makedirs(directory)
fileList = os.listdir(directory)
for fileName in fileList:
    os.remove(directory+"/"+fileName)    

bar = progressbar.ProgressBar()
patient=0
for idx in bar(t1):
    img=io.imread(idx, plugin='simpleitk')
    #print(img.shape)
    for idx1 in range(img.shape[0]):
        #print(img[idx1].shape)
        #print(str(idx1)+"="+str(np.amax(img[idx1])))
        norm= cv2.normalize(img[idx1],img[idx1],alpha=0,beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        io.imsave(directory+'/{}_{}.png'.format(patient, idx1), norm.astype(np.uint8))
    patient=patient+1
