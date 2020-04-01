from time import sleep
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.ndimage.interpolation import rotate
from pprint import pprint
from skimage.transform import radon, iradon


class CT:
    def __init__(self, bitmap_path: str):
        self._bitmap = plt.imread(bitmap_path).astype('float64')[:, :, 0]
        self._sinogram = None

    def radon_transform(self, steps: int):
        w, h = self._bitmap.shape
        sinogram = np.zeros((w, steps), dtype='float64')
        for i, s in enumerate(np.linspace(0, 180, num=steps)):
            rotation = rotate(self._bitmap, s, reshape=False)
            sinogram[:, i] = np.sum(rotation, axis=0)

        self._sinogram = rotate(sinogram, 180, reshape=False)
        return self._sinogram

    def sinogram(self, width=None):
        if width is None:
            width, _ = self._bitmap.shape
        plt.imshow(self.radon_transform(width), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def iradon(self):
        plt.imshow(iradon(self._sinogram), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()


class Radon:
    def __init__(self, bitmap_path: str, da: float, detectors_no: int, span: float):  # da, span in radians
        self._bitmap = plt.imread(bitmap_path).astype('float64')[:, :, 0]
        self._h, self._w = self._bitmap.shape
        self._sinogram = None
        self._center = np.array((self._h - 1, self._w - 1)) / 2
        self._da = da
        self._steps = int(np.pi * 2 / da)
        self._initial_emitter_vector = np.array((0, (self._w - 1))) / 2
        self._rotation_angle = 0
        self._emitter = self._center + self._initial_emitter_vector
        self._detectors_no = detectors_no
        self._emitter_to_1st_detector = np.pi - (span / 2)
        self._detectors = np.zeros((detectors_no, 2))
        self._angle_between_detectors = span / (detectors_no - 1)
        self._calculate_detectors()
        self._reconstructed_bitmap = None

    def _calculate_detectors(self):
        start_to_detector = self._rotation_angle + self._emitter_to_1st_detector
        mag = np.linalg.norm(self._initial_emitter_vector)
        for i in range(self._detectors_no):
            s = np.sin(start_to_detector)
            c = np.cos(start_to_detector)
            self._detectors[i] = (mag * s, mag * c)
            start_to_detector += self._angle_between_detectors
        self._detectors += self._center

    def _calculate_emitter(self):
        mag = np.linalg.norm(self._initial_emitter_vector)
        s = np.sin(self._rotation_angle)
        c = np.cos(self._rotation_angle)
        self._emitter = (mag * s, mag * c)
        self._emitter += self._center

    def _rotate(self):
        self._rotation_angle += self._da
        self._calculate_emitter()
        self._calculate_detectors()

    # def _brasenham(self, p0: np.array, p1: np.array):
    #     if p1[1] <= p0[1]:
    #         p0, p1 = p1, p0
    #     line = np.polyfit((p0[1], p1[1]), (p0[0], p1[0]), 1)
    #     a = line[0]  # directional coefficient
    #
    #     if p0[0] > p1[0]:
    #         p0[0] = -p0[0]
    #         p1[0] = -p1[0]
    #         minus = True
    #     else:
    #         minus = False
    #
    #     points = []
    #
    #     if -1 < a < 1:  # iterating over x's
    #         y0, x0 = map(int, p0)
    #         y1, x1 = map(int, p1)
    #
    #         m_new = 2 * (y1 - y0)
    #         slope_error_new = m_new - (x1 - x0)
    #
    #         y = y0
    #         for x in range(x0, x1 + 1):
    #             if x >= self._w or y >= self._h:
    #                 print(f"p0: {p0} p1: {p1} x: {x} y: {y}")
    #                 break
    #             points.append((y, x))
    #             slope_error_new = slope_error_new + m_new
    #             if slope_error_new >= 0:
    #                 y += 1
    #                 slope_error_new = slope_error_new - 2 * (x1 - x0)
    #     else:  # iterating over y's
    #         if p1[0] <= p0[0]:
    #             p0, p1 = p1, p0
    #
    #         y0, x0 = map(int, p0)
    #         y1, x1 = map(int, p1)
    #
    #         m_new = 2 * (x1 - x0)
    #         slope_error_new = m_new - (y1 - y0)
    #
    #         x = x0
    #         for y in range(y0, y1 + 1):
    #             if x >= self._w or y >= self._h:
    #                 print(f"p0: {p0} p1: {p1} x: {x} y: {y}")
    #                 break
    #             points.append((y, x))
    #             slope_error_new = slope_error_new + m_new
    #             if slope_error_new >= 0:
    #                 x += 1
    #                 slope_error_new = slope_error_new - 2 * (y1 - y0)
    #
    #     points = np.array(points)
    #     if minus:
    #         points = points * np.array((-1, 1))
    #
    #     # y, x = zip(*points)
    #     # plt.scatter(x, y)
    #     # plt.xticks(range(0, 400, 100))
    #     # plt.yticks(range(0, 400, 100))
    #     # plt.show()
    #
    #     return points

    def _brasenham(self, p0, p1):
        y0, x0 = map(int, np.round(p0))
        y1, x1 = map(int, np.round(p1))

        dx = x1 - x0
        dy = y1 - y0

        xsign = dx > 0 or -1
        ysign = dy > 0 or -1

        dx = abs(dx)
        dy = abs(dy)

        if dx > dy:
            xx, xy, yx, yy = xsign, 0, 0, ysign
        else:
            dx, dy = dy, dx
            xx, xy, yx, yy = 0, ysign, xsign, 0

        D = 2 * dy - dx
        y = 0

        for x in range(dx + 1):
            yield x0 + x * xx + y * yx, y0 + x * xy + y * yy
            if D >= 0:
                y += 1
                D -= 2 * dx
            D += 2 * dy

    def _sinogram_step(self, step: int, anim: bool = False):
        self._rotate()
        for i, d in enumerate(self._detectors):
            self._sinogram[i][step] = sum((self._bitmap[y][x] for y, x in self._brasenham(self._emitter, d)))

        if anim:
            norm = np.linalg.norm(self._sinogram)
            return self._sinogram / norm * 255.0

    def sinogram(self):
        self._sinogram = np.zeros((self._detectors_no, self._steps), dtype='float64')
        for i in range(self._steps):
            self._sinogram_step(i)

        norm = np.linalg.norm(self._sinogram)
        self._sinogram = self._sinogram / norm * 255.0
        return self._sinogram

    def sinogram_animated(self):
        self._sinogram = np.zeros((self._detectors_no, self._steps), dtype='float64')
        fig = plt.figure()

        a = self._sinogram_step(0, anim=True)
        im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1, cmap='gray')

        def animate_func(i):
            im.set_array(r._sinogram_step(i, anim=True))
            return [im]

        ani = anim.FuncAnimation(
            fig,
            animate_func,
            frames=range(1, self._steps),
            interval=1
        )
        plt.show()

    def show_sinogram(self):
        plt.imshow(self._sinogram, cmap='gray')
        plt.show()

    def _reset(self):
        self._rotation_angle = 0
        self._emitter = self._center + self._initial_emitter_vector
        self._calculate_detectors()

    def _reconstruction_step(self, step: int, anim: bool = False):
        self._rotate()
        for i, d in enumerate(self._detectors):
            line = self._brasenham(self._emitter, d)
            val = self._sinogram[i][step]
            for y, x in line:
                self._reconstructed_bitmap[y][x] += val

        if anim:
            print(step)
            norm = np.linalg.norm(self._reconstructed_bitmap)
            return self._reconstructed_bitmap / norm * 255.0

    def reconstruct(self):
        self._reset()
        self._reconstructed_bitmap = np.zeros((self._h, self._w), dtype='float64')
        for i in range(self._steps):
            self._reconstruction_step(i)

        norm = np.linalg.norm(self._reconstructed_bitmap)
        self._reconstructed_bitmap = self._reconstructed_bitmap / norm * 255.0

        return self._reconstructed_bitmap

    def reconstruction_animated(self):
        self._reset()
        self._reconstructed_bitmap = np.zeros((self._h, self._w), dtype='float64')
        fig = plt.figure()

        a = self._reconstruction_step(0, anim=True)
        im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1, cmap='gray')

        def animate_func(i):
            im.set_array(r._reconstruction_step(i, anim=True))
            return [im]

        ani = anim.FuncAnimation(
            fig,
            animate_func,
            frames=range(1, self._steps),
            interval=1,
            repeat=False
        )
        plt.show()

    def show_reconstruction(self):
        plt.imshow(self._reconstructed_bitmap, cmap='gray')
        plt.show()


if __name__ == '__main__':
    # ct = CT('/home/prance/PycharmProjects/IwM/CT/images/Shepp_logan.jpg')
    # ct.radon_transform(1000)
    # ct.iradon()
    r = Radon('/home/prance/PycharmProjects/IwM/CT/images/Paski2.jpg', np.pi / 360, 200, *np.pi)
    r.sinogram()
    r.reconstruct()
    r.show_reconstruction()
    # r.sinogram()
    # r.show_sinogram()
    # r.reconstruct()
    # r.show_reconstruction()
