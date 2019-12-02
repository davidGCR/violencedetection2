# #---------Imports
# from numpy import arange, sin, pi
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.figure import Figure
# import tkinter
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib.backends.backend_tkagg import (
#     FigureCanvasTkAgg, NavigationToolbar2Tk)
# # Implement the default Matplotlib key bindings.
# from matplotlib.backend_bases import key_press_handler
# #---------End of imports

# from PIL import Image
# fig = plt.figure())
#     return np.array(image)
# ims = []
# for i in range(60):
#     # x += np.pi / 15.
#     # y += np.pi / 20.
#     im = plt.imshow(f('YOLOv3/data/samples/frame805.jpg'), animated=True)
#     ims.append([im])


# def data_gen():
#     t = data_gen.t
#     cnt = 0
#     while cnt < 1000:
#         cnt+=1
#         t += 0.05
#         y1 = np.sin(2*np.pi*t) * np.exp(-t/10.)
#         y2 = np.cos(2*np.pi*t) * np.exp(-t/10.)
#         # adapted the data generator to yield both sin and cos
#         yield t, y1, y2

# data_gen.t = 0

# # create a figure with two subplots
# fig, (ax1, ax2) = plt.subplots(2,1)

# # intialize two line objects (one in each axes)
# line1, = ax1.plot([], [], lw=2)
# line2, = ax2.plot([], [], lw=2, color='r')
# line = [line1, line2]

# # the same axes initalizations as before (just now we do it for both of them)
# for ax in [ax1, ax2]:
#     ax.set_ylim(-1.1, 1.1)
#     ax.set_xlim(0, 5)
#     ax.grid()

# # initialize the data arrays 
# xdata, y1data, y2data = [], [], []
# def run(data):
#     # update the data
#     t, y1, y2 = data
#     xdata.append(t)
#     y1data.append(y1)
#     y2data.append(y2)

#     # axis limits checking. Same as before, just for both axes
#     for ax in [ax1, ax2]:
#         xmin, xmax = ax.get_xlim()
#         if t >= xmax:
#             ax.set_xlim(xmin, 2*xmax)
#             ax.figure.canvas.draw()

#     # update the data of both line objects
#     line[0].set_data(xdata, y1data)
#     line[1].set_data(xdata, y2data)

#     return line

# ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=10,
#     repeat=False)
# plt.show()
# def f(path='YOLOv3/data/samples/frame805.jpg'):
#     image = Image.open(path)
#     return np.array(image)
# ims = []
# for i in range(60):
#     # x += np.pi / 15.
#     # y += np.pi / 20.
#     im = plt.imshow(f('YOLOv3/data/samples/frame805.jpg'), animated=True)
#     ims.append([im])


# def data_gen():
#     t = data_gen.t
#     cnt = 0
#     while cnt < 1000:
#         cnt+=1
#         t += 0.05
#         y1 = np.sin(2*np.pi*t) * np.exp(-t/10.)
#         y2 = np.cos(2*np.pi*t) * np.exp(-t/10.)
#         # adapted the data generator to yield both sin and cos
#         yield t, y1, y2

# data_gen.t = 0

# # create a figure with two subplots
# fig, (ax1, ax2) = plt.subplots(2,1)

# # intialize two line objects (one in each axes)
# line1, = ax1.plot([], [], lw=2)
# line2, = ax2.plot([], [], lw=2, color='r')
# line = [line1, line2]

# # the same axes initalizations as before (just now we do it for both of them)
# for ax in [ax1, ax2]:
#     ax.set_ylim(-1.1, 1.1)
#     ax.set_xlim(0, 5)
#     ax.grid()

# # initialize the data arrays 
# xdata, y1data, y2data = [], [], []
# def run(data):
#     # update the data
#     t, y1, y2 = data
#     xdata.append(t)
#     y1data.append(y1)
#     y2data.append(y2)

#     # axis limits checking. Same as before, just for both axes
#     for ax in [ax1, ax2]:
#         xmin, xmax = ax.get_xlim()
#         if t >= xmax:
#             ax.set_xlim(xmin, 2*xmax)
#             ax.figure.canvas.draw()

#     # update the data of both line objects
#     line[0].set_data(xdata, y1data)
#     line[1].set_data(xdata, y2data)

#     return line

# ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=10,
#     repeat=False)
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class MyAnimation:
    def __init__(self, images):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 4)
        axes = [ax1, ax2, ax3]
        for idx,img in enumerate(images):
            axes[idx].imshow(img)

        # self.t = np.linspace(0, 80, 400)
        # self.x = np.cos(2 * np.pi * self.t / 10.)
        # self.y = np.sin(2 * np.pi * self.t / 10.)
        # self.z = 10 * self.t

        # ax1.set_xlabel('x')
        # ax1.set_ylabel('y')
        # self.line1 = Line2D([], [], color='black')
        # self.line1a = Line2D([], [], color='red', linewidth=2)
        # self.line1e = Line2D(
        #     [], [], color='red', marker='o', markeredgecolor='r')
        # ax1.add_line(self.line1)
        # ax1.add_line(self.line1a)
        # ax1.add_line(self.line1e)
        # ax1.set_xlim(-1, 1)
        # ax1.set_ylim(-2, 2)
        # ax1.set_aspect('equal', 'datalim')

        # ax2.set_xlabel('y')
        # ax2.set_ylabel('z')
        # self.line2 = Line2D([], [], color='black')
        # self.line2a = Line2D([], [], color='red', linewidth=2)
        # self.line2e = Line2D(
        #     [], [], color='red', marker='o', markeredgecolor='r')
        # ax2.add_line(self.line2)
        # ax2.add_line(self.line2a)
        # ax2.add_line(self.line2e)
        # ax2.set_xlim(-1, 1)
        # ax2.set_ylim(0, 800)

        # ax3.set_xlabel('x')
        # ax3.set_ylabel('z')
        # self.line3 = Line2D([], [], color='black')
        # self.line3a = Line2D([], [], color='red', linewidth=2)
        # self.line3e = Line2D(
        #     [], [], color='red', marker='o', markeredgecolor='r')
        # ax3.add_line(self.line3)
        # ax3.add_line(self.line3a)
        # ax3.add_line(self.line3e)
        # ax3.set_xlim(-1, 1)
        # ax3.set_ylim(0, 800)

        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    # def _draw_frame(self, framedata):
    #     i = framedata
    #     head = i - 1
    #     head_slice = (self.t > self.t[i] - 1.0) & (self.t < self.t[i])

    #     self.line1.set_data(self.x[:i], self.y[:i])
    #     self.line1a.set_data(self.x[head_slice], self.y[head_slice])
    #     self.line1e.set_data(self.x[head], self.y[head])

    #     self.line2.set_data(self.y[:i], self.z[:i])
    #     self.line2a.set_data(self.y[head_slice], self.z[head_slice])
    #     self.line2e.set_data(self.y[head], self.z[head])

    #     self.line3.set_data(self.x[:i], self.z[:i])
    #     self.line3a.set_data(self.x[head_slice], self.z[head_slice])
    #     self.line3e.set_data(self.x[head], self.z[head])

    #     self._drawn_artists = [self.line1, self.line1a, self.line1e,
    #                            self.line2, self.line2a, self.line2e,
    #                            self.line3, self.line3a, self.line3e]

    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _init_draw(self):
        lines = [self.line1, self.line1a, self.line1e,
                 self.line2, self.line2a, self.line2e,
                 self.line3, self.line3a, self.line3e]
        for l in lines:
            l.set_data([], [])

# ani = SubplotAnimation()
# # ani.save('test_sub.mp4')
# plt.show()