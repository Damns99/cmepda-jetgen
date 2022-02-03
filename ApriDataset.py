import h5py
import numpy as np
import matplotlib.pyplot as plt
import os.path

# let's open the file
fileIN = os.path.expanduser('~/Documents/CMEPDA/CMEPDA_exam/Data/jetImage_7_100p_30000_40000.h5')
f = h5py.File(fileIN)
# and see what it contains
print(list(f.keys()))

print('particleFeatureNames')
particleFeatureNames = np.array(f.get('particleFeatureNames'))
print(particleFeatureNames)

jetConstituentList = np.array(f.get('jetConstituentList'))
print('jetConstituentList')
print(np.shape(jetConstituentList))
print(particleFeatureNames[np.array([5,6,7,8,10,11])])
print(jetConstituentList[0,0:5,np.array([5,6,7,8,10,11])])

jetImages = f.get('jetImage')
print('jetImage')
print(np.shape(jetImages))
#plt.imshow(jetImages[152])

print('jetFeatureNames')
print(np.array(f.get('jetFeatureNames')))

jets = f.get('jets')
print('jets')
print(np.shape(jets))

##
# explore particle features
etarel = jetConstituentList[:,:,8]
print(np.max(etarel))
print(np.min(etarel))
plt.figure()
plt.hist(etarel[0, :], 20, [-0.5, 0.5])
phirel = jetConstituentList[:,:,11]
print(np.max(phirel))
print(np.min(phirel))
plt.figure()
plt.hist(phirel[0, :], 20, [-0.5, 0.5])

# create 100 x 100 pt images from -2 to 2 in etarel and from -1 to 1 in phirel
pt = jetConstituentList[:,:,5]
print(np.shape(pt))
print(np.max(np.log(pt)))
print(np.min(np.log(pt)))
plt.figure()
plt.hist(np.log(pt[0, :]), 20, [0, 5])

def histogram2d(etarel, phirel, pt):
    newjetImage = np.histogram2d(etarel, phirel, [100,100], [[-2,2],[-1,1]], weights = pt)
    return newjetImage[0]
newjetImages = np.array([])
for i, j, k in zip(etarel[0:1,:], phirel[0:1,:], pt[0:1,:]):
    print([i,j,k])
    newjetImages = np.append(newjetImages, np.histogram2d(i, j, [100,100], [[-2,2],[-1,1]], weights = k))

print(np.shape(newjetImages))
#plt.figure()
#plt.imshow(newjetImages[2])





plt.show()