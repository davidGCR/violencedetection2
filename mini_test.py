import torch
import numpy as np
from LOCALIZATION.bounding_box import BoundingBox
from LOCALIZATION.point import Point

# class Person(object):
#     def __init__(self, name, ssn):
#         self.name = name
#         self.ssn = ssn

#     def __eq__(self, other):
#         return isinstance(other, Person) and self.ssn == other.ssn and  self.name == other.name

#     def __hash__(self):
#         # use the hashcode of self.ssn since that is used
#         # for equality checks as well
#         return hash((self.ssn, self.name))

# result = []
# # p = Person('Foo Bar', 123456789)
# # q = Person('Foo Bar', 123456789)
# p = BoundingBox(Point(5, 5), Point(24, 24))
# q = BoundingBox(Point(5.3432, 6), Point(24.98, 24.87))
# r = BoundingBox(Point(5.34320,6.0), Point(24.980000,24.8700))
# result.append(p)
# result.append(q)
# result.append(r)
# result = set(result)
# # result = list(result)
# print(len(result)) # len = 2
# for r in result:
#     print(r)

import numpy as np
import itertools
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# fig = plt.figure()


# def f(x, y):
#     return np.sin(x) + np.cos(y)

# x = np.linspace(0, 2 * np.pi, 120)
# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

# im = plt.imshow(f(x, y), animated=True)


# def updatefig(*args):
#     global x, y
#     x += np.pi / 15.
#     y += np.pi / 20.
#     im.set_array(f(x, y))
#     return im,

# ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
# plt.show()
# aa = [Point(1,2), Point(17,27), Point(16,26)]

# for a, b in itertools.combinations(aa, 2):
#     print(a,b)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
fig = plt.figure()
def f(path='YOLOv3/data/samples/frame805.jpg'):
    image = Image.open(path)
    return np.array(image)

# x = np.linspace(0, 2 * np.pi, 120)
# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(60):
    # x += np.pi / 15.
    # y += np.pi / 20.
    im = plt.imshow(f('YOLOv3/data/samples/frame805.jpg'), animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# fig = plt.figure()


# def f(x, y):
#     return np.sin(x) + np.cos(y)

# x = np.linspace(0, 2 * np.pi, 120)
# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
# # ims is a list of lists, each row is a list of artists to draw in the
# # current frame; here we are just animating one artist, the image, in
# # each frame
# ims = []
# for i in range(60):
#     x += np.pi / 15.
#     y += np.pi / 20.
#     im = plt.imshow(f(x, y), animated=True)
#     ims.append([im])

# ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
#                                 repeat_delay=1000)

# plt.show()