import numpy as np
import random
import os
from skimage import io
from glob import glob
import progressbar
bar = progressbar.ProgressBar()
def findPatches(patchesNumber,classLabel):
    h=31
    w=31
    ct=0
    while ct < patchesNumber:
        T1Images = glob('T1Images//**')
        T1 = random.choice(T1Images)
        fn = os.path.basename(T1)
        #print("File name is "+str(fn))
        label = io.imread('Labels//' + fn[:-4] + '.png')
        if len(np.argwhere(label == classLabel)) < 300:
                    continue
        img = io.imread(T1).reshape(1,240, 240)
        p = random.choice(np.argwhere(label == classLabel))
        label=label.reshape(1,240, 240) 
        p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
        patch = np.array([i[int(p_ix[0]+1):int(p_ix[1]), int(p_ix[2]+1):int(p_ix[3])] for i in img])
        labelPatch = np.array([i[int(p_ix[0]+1):int(p_ix[1]), int(p_ix[2]+1):int(p_ix[3])] for i in label])
        #print("Made")
        if patch.shape != (1, h, w)  and label.shape != (1, h, w) or len(np.argwhere(patch == 0)) == (h * w):
                continue  
        path = 'T1Patches'
        patch = patch.reshape(h,w)
        if len(np.argwhere(labelPatch == classLabel)) < 480:
                    continue
        #print(len(np.argwhere(patch == classLabel)))
        io.imsave(path + '//{}_{}.png'.format(classLabel, ct), patch)
        path = 'LabelPatches'
        labelPatch = labelPatch.reshape(h,w)
        io.imsave(path + '//{}_{}.png'.format(classLabel, ct), labelPatch)
        ct += 1
    
    
def makePatches(patchesNumber):
    classes=[0,1,2,3,4]
    #gt = glob('../HGG/brats*/*more.*/*.mha')
    directory='T1Patches'
    if not os.path.exists(directory):
        os.makedirs(directory)
    fileList = os.listdir(directory)
    for fileName in fileList:
        os.remove(directory+"/"+fileName)
    directory='LabelPatches'
    if not os.path.exists(directory):
        os.makedirs(directory)
    fileList = os.listdir(directory)
    for fileName in fileList:
        os.remove(directory+"/"+fileName) 
        
        
    for idx_class in bar(range(5)):
        findPatches(patchesNumber,idx_class)
        #print(classes[idx_class])

#np.set_printoptions(threshold=np.nan)
#T1Images="T1Images"
#T1Images = glob('T1Images//**')
#GTImages = glob('Labels//**')
#im_path = random.choice(T1Images)
#fn = os.path.basename(im_path)
#print(fn[:-4])
#print(im_path)
#print(fn)
#img=io.imread("Labels//0_146.png")
#img=io.imread(im_path)
#if len(np.argwhere(label == class_num)) < 10:
#    continue
#patch=random.choice(np.argwhere(img==4))
#print(patch)

#patchesNumber = int(input("Enter the number of patches for each class : "))
#print(patchesNumber)
#for idx_class in bar(range(5)):
#    for idx in range(patchesNumber):
        #print(idx)
patchesNumber = int(input("Enter the number of patches for each class : "))
makePatches(patchesNumber)