from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from DICOMhandler import DICOMhandler
from scipy.ndimage.filters import convolve
from ctypes import *
import csv
from multiprocessing import Process


class Radon:
    brasenham_lib = CDLL("/home/prance/PycharmProjects/IwM/CT/brasenham.so")

    def __init__(self, bitmap_path: str, da: float, detectors_no: int, span: float,
                 dicom: bool = False):  # da, span in radians

        if dicom:
            self._dicom = DICOMhandler().load(bitmap_path)
            self._bitmap = self._dicom.bitmap
        else:
            self._bitmap = plt.imread(bitmap_path).astype('float64')
            if len(self._bitmap.shape) == 3:
                self._bitmap = self._bitmap[:, :, 0]

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
        self._c_array_type = c_bool * (self._h * self._w)

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
        result = self._c_array_type()
        y0, x0 = map(int, np.round(p0))
        y1, x1 = map(int, np.round(p1))
        self.brasenham_lib.brasenham(self._h, self._w, y0, x0, y1, x1, byref(result))

        result = np.array(result)
        result.shape = self._h, self._w

        return result

    def _sinogram_step(self, step: int, anim: bool = False):
        self._rotate()
        for i, d in enumerate(self._detectors):
            self._sinogram[i][step] = np.mean(self._bitmap[self._brasenham(self._emitter, d)])

        if anim:
            norm = self._sinogram.max()
            return self._sinogram / norm * 255.0

    def sinogram(self):
        self._sinogram = np.zeros((self._detectors_no, self._steps), dtype='float64')
        for i in range(self._steps):
            self._sinogram_step(i)

        norm = self._sinogram.max()
        self._sinogram = self._sinogram / norm * 255.0
        return self._sinogram

    def sinogram_animated(self):
        self._sinogram = np.zeros((self._detectors_no, self._steps), dtype='float64')
        fig = plt.figure()

        a = self._sinogram_step(0, anim=True)
        im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1, cmap='gray')

        def animate_func(i):
            im.set_array(self._sinogram_step(i, anim=True))
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
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def _reset(self):
        self._rotation_angle = 0
        self._emitter = self._center + self._initial_emitter_vector
        self._calculate_detectors()

    def _reconstruction_step(self, step: int, anim: bool = False):
        self._rotate()
        for i, d in enumerate(self._detectors):
            line = self._brasenham(self._emitter, d)
            self._reconstructed_unnormed[line] += self._sinogram[i][step]

        if anim:
            print(step)
            norm = self._reconstructed_unnormed.max()
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
        self._reconstructed_bitmap = self._reconstructed_unnormed * 255.0 / self._reconstructed_unnormed.max()

    def convolve(self, k=100, mode='constant'):
        k = np.array([[10, 10, 10],
                      [10, k, 10],
                      [10, 10, 10]])
        self._reconstructed_unnormed = convolve(self._reconstructed_unnormed, k, mode=mode)

        return self._reconstructed_unnormed

    def reconstruction_animated(self):
        self._reset()
        self._reconstructed_unnormed = np.zeros((self._h, self._w), dtype='float64')
        fig = plt.figure()

        a = self._reconstruction_step(0, anim=True)
        im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1, cmap='gray')

        def animate_func(i):
            im.set_array(self._reconstruction_step(i, anim=True))
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
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def show_difference(self):
        diff = self._bitmap - self._reconstructed_bitmap
        diff += 255
        diff = diff * 255.0 / diff.max()
        plt.imshow(diff, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()


class GUI:
    def __openAndResize(self, filename, height, panel):
        image = Image.open(filename)
        image = image.resize((500, height), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)
        panel.configure(image=image)
        panel.image = image

    def _showBasicImage(self):
        self._filename = filedialog.askopenfilename()
        self.__openAndResize(self._filename, 480, self.panel)

    def _makeSinogram(self):
        self._r = Radon(self._filename, np.pi / 360, 200, np.pi)
        bitmap = self._r.sinogram()
        plt.imsave('outfile.jpg', bitmap, cmap='gray')
        self.__openAndResize("outfile.jpg", bitmap.shape[0], self.panel_sinogram)

    def _showResult(self):
        bitmap = self._r.reconstruct()
        plt.imsave('result.jpg', bitmap, cmap='gray')
        self.__openAndResize("result.jpg", 480, self.panel_result)

    def __init__(self):
        self._filename = "";

        self.root = Tk()
        self.root.geometry("1500x500")

        self.f = Frame(self.root, height=500, width=500)
        self.f.pack_propagate(0)
        self.f.place(x=0, y=0)

        self.f2 = Frame(self.root, height=500, width=500)
        self.f2.pack_propagate(0)
        self.f2.place(x=500, y=0)

        self.f3 = Frame(self.root, height=500, width=500)
        self.f3.pack_propagate(0)
        self.f3.place(x=1000, y=0)

        self.chooseFileButton = Button(self.f, text="Wybierz plik", command=self._showBasicImage)
        self.chooseFileButton.pack(fill=X, expand=0)

        self.sinogramButton = Button(self.f2, text="Pokaż sinogram", command=self._makeSinogram)
        self.sinogramButton.pack(fill=X, expand=0)

        self.resultButton = Button(self.f3, text="Pokaż wynik końcowy", command=self._showResult)
        self.resultButton.pack(fill=X, expand=0)

        self.panel = Label(self.f)
        self.panel.pack(fill=BOTH, expand=1)

        self.panel_sinogram = Label(self.f2)
        self.panel_sinogram.pack(fill=BOTH, expand=1)

        self.panel_result = Label(self.f3)
        self.panel_result.pack(fill=BOTH, expand=1)

        self.root.mainloop()


class Tests:
    @staticmethod
    def mse(orig, final):
        err = np.sum((orig - final) ** 2)
        err /= orig.shape[0] * orig.shape[1]
        res = np.sqrt(err)
        return res

    def __init__(self):
        self.image_path = "images/Shepp_logan.jpg"

    def test1(self):
        f = open("foff2results1.csv", "w")
        w = csv.writer(f)
        steps = 180
        detectors_span = 1
        for detectors in range(90, 721, 90):
            r = Radon(self.image_path, np.pi / steps, detectors, detectors_span * np.pi)
            r.sinogram()
            r.reconstruct(filter=False)
            print((steps, detectors, detectors_span, self.mse(r._bitmap, r._reconstructed_bitmap)))
            w.writerow((steps, detectors, detectors_span, self.mse(r._bitmap, r._reconstructed_bitmap)))
        f.close()

    def test2(self):
        f = open("foff2results2.csv", "w")
        w = csv.writer(f)
        detectors = 180
        detectors_span = 1
        for steps in range(90, 721, 90):
            r = Radon(self.image_path, np.pi / steps, detectors, detectors_span * np.pi)
            r.sinogram()
            r.reconstruct(filter=False)
            print((steps, detectors, detectors_span, self.mse(r._bitmap, r._reconstructed_bitmap)))
            w.writerow((steps, detectors, detectors_span, self.mse(r._bitmap, r._reconstructed_bitmap)))
        f.close()

    def test3(self):
        f = open("foff2results3.csv", "w")
        w = csv.writer(f)
        detectors = 180
        steps = 180
        for detectors_span in np.linspace(45, 270, 45):
            r = Radon(self.image_path, np.pi / steps, detectors, detectors_span / 180 * np.pi)
            r.sinogram()
            r.reconstruct(filter=False)
            print((steps, detectors, detectors_span, self.mse(r._bitmap, r._reconstructed_bitmap)))
            w.writerow((steps, detectors, detectors_span, self.mse(r._bitmap, r._reconstructed_bitmap)))
        f.close()

    def test4(self):
        f = open("foff2results4.csv", "w")
        w = csv.writer(f)
        detectors = 180
        steps = 180
        detectors_span = 180
        r = Radon(self.image_path, np.pi / steps, detectors, detectors_span / 180 * np.pi)
        r.sinogram()
        r.reconstruct(filter=False)
        print((steps, detectors, detectors_span, self.mse(r._bitmap, r._reconstructed_bitmap)))
        w.writerow((steps, detectors, detectors_span, self.mse(r._bitmap, r._reconstructed_bitmap)))
        r.reconstruct(filter=True)
        print((steps, detectors, detectors_span, self.mse(r._bitmap, r._reconstructed_bitmap)))
        w.writerow((steps, detectors, detectors_span, self.mse(r._bitmap, r._reconstructed_bitmap)))
        f.close()

    def test5(self):
        f = open("2results5.csv", "w")
        w = csv.writer(f)
        detectors = 180
        steps = 180
        detectors_span = 180
        r = Radon(self.image_path, np.pi / steps, detectors, detectors_span / 180 * np.pi)
        r.sinogram()
        r._reconstructed_unnormed = np.zeros((r._h, r._w), dtype='float64')
        for i in range(r._steps):
            rs = r._reconstruction_step(i, anim=True)
            rsme = self.mse(r._bitmap, rs)
            print((i, detectors, detectors_span, rsme))
            w.writerow((i, detectors, detectors_span, rsme))
        f.close()


if __name__ == '__main__':
    # t = Tests()
    # p1 = Process(target=t.test1)
    # p2 = Process(target=t.test2)
    # p3 = Process(target=t.test3)
    # p1.start()
    # p2.start()
    # p3.start()
    # p1.join()
    # p2.join()
    # p3.join()

    r = Radon("images/Shepp_logan.jpg", np.pi/720, 500, 3*np.pi/2)
    r.sinogram()
    r.show_sinogram()
    r.reconstruct()
    r.show_reconstruction()
