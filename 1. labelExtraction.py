from glob import glob
from skimage import io
import numpy as np
np.set_printoptions(threshold=np.nan)
import os

import progressbar


gt = glob('../HGG/brats*/*more.*/*.mha')
directory='Labels'
if not os.path.exists(directory):
    os.makedirs(directory)
fileList = os.listdir(directory)
for fileName in fileList:
    os.remove(directory+"/"+fileName)    
#print("Done") 
bar = progressbar.ProgressBar()
patient=0
for idx in bar(gt):
    print("Name "+str(idx))
    img=io.imread(idx, plugin='simpleitk')
    #print(img.shape)
    for idx1 in range(img.shape[0]):
        #print("Inside")
        #print(img[idx1].shape)
        #print(str(idx1)+"="+str(np.amax(img[idx1])))
        #norm= cv2.normalize(img[idx1],img[idx1],alpha=0,beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #print(img[idx1])
        io.imsave(directory+'/{}_{}.png'.format(patient, idx1), img[idx1])
    patient=patient+1
