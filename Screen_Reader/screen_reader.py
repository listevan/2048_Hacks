from PIL import Image, ImageGrab 
from matplotlib import pyplot as plt
import numpy as np

def compare_color (a, b, delta = 7):
    return abs(a[0] - b[0]) < delta and abs(a[1] - b[1]) < delta and abs(a[2] - b[2]) < delta

class Screen_Reader:
    color_dict = {
        (201, 193, 181) : 0,
        (236, 228, 219) : 2,
        (237, 223, 200) : 4,
        (243, 178, 121) : 8,
        (246, 149, 99) : 16,
        (247, 124, 95) : 32,
        (247, 95, 59) : 64,
        (237, 207, 114) : 128,
        (237, 204, 97) : 256,
        (237, 199, 80) : 512,
        (237, 196, 63) : 1024,
        (237, 194, 45) : 2048,
    }
    colors = color_dict.keys()
    numbers = color_dict.values()

    back_color = (250, 248, 239)
    wall_color = (187, 172, 159)
    number_color = (118, 109, 101)

    def __init__(self, dir = None):
        if dir:
            self.image = np.asarray(Image.open(dir), dtype=np.int32)[:, :, :3]
        else:
            self.image = np.asarray(ImageGrab.grab(bbox = None), dtype=np.int32)[:, :, :3]

        self.top_left = [50, 50]
        found = False
        step = 5
        for j in range(0, self.image.shape[1] - step, step):
            for i in range(0, self.image.shape[0] - step, step):
                for ii in range(i, i+step):
                    for jj in range(j, j+step):
                        found = compare_color(self.image[ii, jj], Screen_Reader.wall_color)
                        if (not found):
                            break
                    if (not found):
                        break
                if (found):
                    self.top_left = [i, j]
                    break
            if (found):
                break

        self.top_right = self.top_left.copy()
        self.bottom_left = self.top_left.copy()

        while compare_color(self.image[self.top_right[0], self.top_right[1]], Screen_Reader.wall_color):
            self.top_right[1]+=1

        while compare_color(self.image[self.bottom_left[0], self.bottom_left[1]], Screen_Reader.wall_color):
            self.bottom_left[0]+=1

        self.width = self.top_right[1] - self.top_left[1]
        self.height = self.bottom_left[0] - self.top_left[0]

        self.w_step = int(self.width / 4)
        self.h_step = int(self.height / 4)
        self.arr = np.zeros((4, 4))
    
    def getArray(self, display = False):
        if display:
            plt.figure()
        for ii in range(4):
            for jj in range(4):
                if display:
                    plt.subplot(4, 4, (4 * ii) + jj + 1)

                i = self.top_left[0] + (self.h_step * ii)
                j = self.top_left[1] + (self.w_step * jj)

                if display:
                    plt.imshow(self.image[i:i+self.h_step, j:j+self.w_step])
                    plt.title(("at " + str(i) + ", " + str(j)))

                r = self.image[i:i+self.h_step, j:j+self.w_step, 0]
                g = self.image[i:i+self.h_step, j:j+self.w_step, 1]
                b = self.image[i:i+self.h_step, j:j+self.w_step, 2]

                rs = sorted([(x , ((r == x).sum())) for x in np.unique(r)], key = lambda x : x[1], reverse=True)
                gs = sorted([(x , ((g == x).sum())) for x in np.unique(g)], key = lambda x : x[1], reverse=True)
                bs = sorted([(x , ((b == x).sum())) for x in np.unique(b)], key = lambda x : x[1], reverse=True)

                curr = (rs[0][0], gs[0][0], bs[0][0])

                for color in Screen_Reader.colors:
                    if (compare_color(curr, color, 20)):
                        curr = color
                        break

                self.arr[int((i - self.top_left[0]) / (self.h_step)), int((j - self.top_left[1]) / (self.w_step))] = Screen_Reader.color_dict[curr]

        return self.arr

    def update(self, dir = None):
        if dir:
            self.image = np.asarray(Image.open(dir), dtype=np.int32)[:, :, :3]
        else:
            self.image = np.asarray(ImageGrab.grab(bbox = None), dtype=np.int32)[:, :, :3]

        return self.getArray(self)