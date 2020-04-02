from time import sleep
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.ndimage.interpolation import rotate
from pprint import pprint
from skimage.transform import radon, iradon
from DICOMhandler import DICOMhandler
from scipy.ndimage.filters import convolve


class Radon:
    def __init__(self, bitmap_path: str, da: float, detectors_no: int, span: float,
                 dicom: bool = False):  # da, span in radians
        if dicom:
            self._dicom = DICOMhandler().load(bitmap_path)
            self._bitmap = self._dicom.bitmap
        else:
            self._bitmap = plt.imread(bitmap_path).astype('float64')[:, :, 0]

        self._h, self._w = self._bitmap.shape
        self._sinogram = None
        self._center = np.array((self._h - 1, self._w - 1)) / 2
        self._da = da
        self._steps = int(2 * np.pi / da)
        self._initial_emitter_vector = np.array((0, (self._w - 1))) / 2
        self._rotation_angle = 0
        self._emitter = self._center + self._initial_emitter_vector
        self._detectors_no = detectors_no
        self._emitter_to_1st_detector = np.pi - (span / 2)
        self._detectors = np.zeros((detectors_no, 2))
        self._angle_between_detectors = span / (detectors_no - 1)
        self._calculate_detectors()
        self._reconstructed_bitmap = None
        self._reconstructed_unnormed = None

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
        x, y = 0, 0

        while True:
            xr = x0 + x * xx + y * yx
            yr = y0 + x * xy + y * yy
            if xr < 0 or xr >= self._w or yr < 0 or yr >= self._h:
                break
            yield yr, xr
            if D >= 0:
                y += 1
                D -= 2 * dx
            D += 2 * dy
            x += 1

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
                self._reconstructed_unnormed[y][x] += val

        if anim:
            print(step)
            norm = np.linalg.norm(self._reconstructed_unnormed)
            return self._reconstructed_unnormed / norm * 255.0

    def reconstruct(self, filter=True):
        self._reset()
        self._reconstructed_unnormed = np.zeros((self._h, self._w), dtype='float64')
        for i in range(self._steps):
            self._reconstruction_step(i)

        if filter:
            self.convolve()

        self._normalize()
        return self._reconstructed_bitmap

    def _normalize(self):
        norm = np.linalg.norm(self._reconstructed_unnormed)
        self._reconstructed_bitmap = self._reconstructed_unnormed / norm * 255.0

    def convolve(self, k=100, mode='constant'):
        k = np.array([[10, 10, 10],
                      [10, k, 10],
                      [10, 10, 10]])
        self._reconstructed_unnormed = convolve(self._reconstructed_unnormed, k, mode=m)

        return self._reconstructed_unnormed

    def reconstruction_animated(self):
        self._reset()
        self._reconstructed_unnormed = np.zeros((self._h, self._w), dtype='float64')
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

    def show_difference(self):
        diff = self._reconstructed_bitmap - self._bitmap
        # norm = np.linalg.norm(diff)
        # diff = diff / norm * 255.0
        plt.imshow(diff, cmap='gray')
        plt.show()


if __name__ == '__main__':
    # ct = CT('/home/prance/PycharmProjects/IwM/CT/images/Shepp_logan.jpg')
    # ct.radon_transform(1000)
    # ct.iradon()
    r = Radon('/home/prance/PycharmProjects/IwM/CT/images/Shepp_logan.jpg', np.pi / 720, 400, 3 * np.pi / 2)

    s = r.sinogram()
    # r.show_sinogram()
    # r.reconstruction_animated()
    r.reconstruct()
    r.show_reconstruction()
    # r.convolve()
    # r.show_reconstruction()
    # r = plt.imread('/home/prance/PycharmProjects/IwM/CT/images/marchewka.jpg').astype('float64')[:, :, 1]
    # plt.imshow(r, cmap='gray')
    # plt.imsave('/home/prance/PycharmProjects/IwM/CT/images/marchewka.jpg', r, cmap='gray')
    # r.show_difference()

    # plt.imshow(r.iradon_transform(s, theta=np.linspace(0, 179, 180)), cmap='gray')
    # plt.show()
    # r.sinogram()
    # r.show_sinogram()
    # r.reconstruct()
    # r.show_reconstruction()
