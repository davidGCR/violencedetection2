# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython

# %%
import numpy as np
import cv2
import os
from  matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
# import ipyvolume as ipv
# get_ipython().run_line_magic('matplotlib', 'inline')


folder = '/Users/davidchoqueluqueroman/Desktop/PROJECTS-SOURCE-CODES/violencedetection2/DATASETS'
im1=cv2.imread(os.path.join(folder,'RWF-2000/frames/train/Fight/_2RYnSFPD_U_0/frame1.jpg'))
im2=cv2.imread(os.path.join(folder,'RWF-2000/frames/train/Fight/_2RYnSFPD_U_0/frame2.jpg'))

print(im1.shape, im2.shape)


# %%
plt.imshow(im1[:,:,::-1])
plt.show()


# %%
plt.imshow(im2[:,:,::-1])
plt.show()


# %%
hsv = np.zeros_like(im1)
hsv[...,1] = 255

im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

# %%
flow = cv2.calcOpticalFlowFarneback(im1,im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
plt.imshow(bgr[:,:,::-1])
# plt.imshow(hsv)
plt.show()

# pixel_colors = bgr.reshape((np.shape(bgr)[0]*np.shape(bgr)[1], 3))
# norm = colors.Normalize(vmin=-1.,vmax=1.)
# norm.autoscale(pixel_colors)
# pixel_colors = norm(pixel_colors).tolist()

# h, s, v = cv2.split(hsv)
# fig = plt.figure()
# axis = fig.add_subplot(1, 1, 1, projection="3d")

# axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
# axis.set_xlabel("Hue")
# axis.set_ylabel("Saturation")
# axis.set_zlabel("Value")
# plt.show()


# %%
path='/Users/davidchoqueluqueroman/Desktop/PROJECTS-SOURCE-CODES/violencedetection2/IMAGES'
dyn_img = cv2.imread(os.path.join(path,'rwf-2000-3OKArbzg1uc_3-1.png'))
plt.imshow(dyn_img[:,:,::-1])
plt.show()


# %%
dyn_img = cv2.cvtColor(dyn_img, cv2.COLOR_BGR2RGB)
# figure, plots = plt.subplots(ncols=3, nrows=1)
# for i, subplot in zip(range(3), plots):
#     temp = np.zeros(dyn_img.shape, dtype='uint8')
#     temp[:,:,i] = dyn_img[:,:,i]
#     subplot.imshow(temp)
#     subplot.set_axis_off()
# plt.show()

# %%

r, g, b = cv2.split(dyn_img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

pixel_colors = dyn_img.reshape((np.shape(dyn_img)[0]*np.shape(dyn_img)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()

# %%
hsv_img = cv2.cvtColor(dyn_img, cv2.COLOR_RGB2HSV)
hsv_img[...,1] = 255
plt.imshow(hsv_img)
plt.show()

h, s, v = cv2.split(hsv_img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()


bgr2 = cv2.cvtColor(hsv_img,cv2.COLOR_HSV2BGR)
plt.imshow(bgr2[:,:,::-1])
plt.show()



# %%

# N = 1000
# x, y, z = np.random.normal(0, 1, (3, N))
# fig = ipv.figure()
# scatter = ipv.scatter(x, y, z)
# ipv.show()


# %%



