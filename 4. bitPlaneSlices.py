import numpy as np
from skimage import io
from glob import glob
import os
#import progressbar

h=31
w=31
#bar = progressbar.ProgressBar()
T1Images = glob('T1Patches//**')


directory='T1PatchesBitplane'
if not os.path.exists(directory):
    os.makedirs(directory)
fileList = os.listdir(directory)
for fileName in fileList:
    os.remove(directory+"/"+fileName)    

for idx in (T1Images):
    #print(idx)
    fn = os.path.basename(idx)  
    #print(fn)
    im=io.imread(idx)
    First=np.zeros(shape=(h, w),dtype=np.uint8)
    Second=np.zeros(shape=(h, w),dtype=np.uint8)
    Third=np.zeros(shape=(h, w),dtype=np.uint8)
    Fourth=np.zeros(shape=(h, w),dtype=np.uint8)
    Fifth=np.zeros(shape=(h, w),dtype=np.uint8)
    Sixth=np.zeros(shape=(h, w),dtype=np.uint8)
    Seventh=np.zeros(shape=(h, w),dtype=np.uint8)
    Eighth=np.zeros(shape=(h, w),dtype=np.uint8)
    for i in range(0,h):
        for j in range(0,w):
            Val='{0:08b}'.format(int(im[i][j]))
            First[i][j]=int(Val[0])*128
            Second[i][j]=int(Val[1])*64
            Third[i][j]=int(Val[2])*32
            Fourth[i][j]=int(Val[3])*16
            Fifth[i][j]=int(Val[4])*8
            Sixth[i][j]=int(Val[5])*4
            Seventh[i][j]=int(Val[6])*2
            Eighth[i][j]=int(Val[7])*1
            
    temp=np.vstack((First,Second))
    temp=np.vstack((temp,Third))
    temp=np.vstack((temp,Fourth))
    temp=np.vstack((temp,Fifth))  
    temp=np.vstack((temp,Sixth))
    temp=np.vstack((temp,Seventh))
    temp=np.vstack((temp,Eighth))
    #print(temp.shape)
    io.imsave(directory+'//'+str(fn), temp)
         

