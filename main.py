import asyncio
import math
import shutil
from datetime import datetime

from tkinter import *
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
import scipy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.image import imread
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from DICOMhandler import DICOMhandler
from scipy.ndimage.filters import convolve
import matplotlib
matplotlib.use("TkAgg")
import threading
import pygubu
import ttkwidgets


class Radon:

    def __init__(self, bitmap_path: str, da: float, detectors_no: int, span: float,
                 dicom: bool = False):  # da, span in radians

        # if dicom:
        #     self._dicom = DICOMhandler().load(bitmap_path)
        #     self._bitmap = self._dicom.bitmap
        # else:

        self._dicom = None
        self._bitmap = plt.imread(bitmap_path).astype('float64')[:, :, 0]

        self._h, self._w = self._bitmap.shape
        self._sinogram = None
        self._sinogram_canvas = None
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

    def setDICOM(self,filename):
        self._dicom = DICOMhandler().load(filename)
        self._bitmap = self._dicom.bitmap

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

    def sinogram_animated(self, frame, anim_callback):
        self._sinogram = np.zeros((self._detectors_no, self._steps), dtype='float64')
        fig = plt.figure(figsize=(7,3))

        a = self._sinogram_step(0, anim=True)
        im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1, cmap='gray')

        def animate_func(i):
            im.set_data(self._sinogram_step(i, anim=True))

            if (i >= self._steps - 1):
                anim_callback()

            return [im]


        self._sinogram_canvas = FigureCanvasTkAgg(fig, master=frame)
        self._sinogram_canvas.get_tk_widget().pack(side=LEFT, anchor="s")

        ani = anim.FuncAnimation(
            fig,
            animate_func,
            frames= range(1, self._steps),
            interval=1,
            repeat=False
        )

        self._sinogram_canvas.draw()

    def show_sinogram(self,frame,callback):
        fig = plt.figure(figsize=(7,3))

        plt.imshow(self._sinogram, cmap='gray')
        #
        # frame.clear()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack()

        canvas.draw()
        callback()

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
        self._reconstructed_unnormed = convolve(self._reconstructed_unnormed, k, mode=mode)

        return self._reconstructed_unnormed

    def reconstruction_animated(self, frame, anim_callback):
        self._reset()
        self._reconstructed_unnormed = np.zeros((self._h, self._w), dtype='float64')
        fig = plt.figure(figsize=(7,4))

        a = self._reconstruction_step(0, anim=True)
        im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1, cmap='gray')

        def animate_func(i):
            im.set_data(self._reconstruction_step(i, anim=True))

            if (i >= self._steps - 1):
                anim_callback()

            return [im]

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack()

        ani = anim.FuncAnimation(
            fig,
            animate_func,
            frames=range(1, self._steps),
            interval=1,
            repeat=False
        )

        canvas.draw()

    def show_reconstruction(self,frame):
        fig = plt.figure(figsize=(7,4))
        plt.imshow(self._reconstructed_bitmap, cmap='gray')
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def show_difference(self):
        diff = self._reconstructed_bitmap - self._bitmap
        # norm = np.linalg.norm(diff)
        # diff = diff / norm * 255.0
        plt.imshow(diff, cmap='gray')
        plt.show()

class GUI:
    def get_variable(self,name):
        return self.builder.get_variable(name).get()

    @staticmethod
    def open_and_resize(filename,width, height, panel):
        image = Image.open(filename)
        image = image.resize((width,height),Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)
        panel.configure(image=image)
        panel.image = image

    def _show_basic_image(self,event=None):
        if (event):
            self._filename = event.widget.cget('path')
        GUI.open_and_resize(self._filename, 500, 480, self.panel)

        self.show_tab(1)

    def _afterSinogram(self):
        self.show_tab(2)

    def destroy(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def generate_sinogram(self):
        self._r = Radon(self._filename, np.pi / self.slider_alpha.get(), self.slider_detectors.get(), np.pi / 180 * self.slider_range.get())

        self.destroy(self.f2)

        if ("ANIMACJA" in self.typ.get()):
            self._r.sinogram_animated(self.f2,self._afterSinogram)
        else:
            self._r.sinogram()
            self._r.show_sinogram(self.f2,self._afterSinogram)

    def create_dicom(self):
        dh = DICOMhandler()

        p = DICOMhandler.Patient(self.get_variable('pesel'),
                                 self.get_variable('name'),
                                 self.get_variable('pesel')[0:5],
                                 self.get_variable('sex'))

        date = self.builder.get_object('calendar_input').selection
        comments = self.text_comments.get("1.0", END)

        metadata = {
            "patient": p,
            'date': date,
            'comments': comments
        }

        self.dicom_path = p.id + ".DCM"

        dh.new(self.dicom_path, self._r._reconstructed_bitmap, metadata)

        print("Ok")

        o = dh.load(self.dicom_path)

        self.show_dicom(o)

    def show_dicom(self, o):
        fig = plt.figure(figsize=(7, 5))
        plt.imshow(o.bitmap, cmap='gray')
        canvas = FigureCanvasTkAgg(fig, master=self.f4)
        canvas.get_tk_widget().pack()
        canvas.draw()
        self.show_tab(3)

    def save_file(self):
        shutil.copy(self.dicom_path,self.get_variable("save_file_path"))

    def generate_result(self):
        self.destroy(self.f3)

        if ("Animacja" in self.get_variable("result_type")):
            self._r.reconstruction_animated(self.f3, self._afterSinogram)
        elif ("bez" in self.get_variable("result_type")):
            self._r.reconstruct(filter=False)
            self._r.show_reconstruction(self.f3)
        else:
            self._r.reconstruct(filter=True)
            self._r.show_reconstruction(self.f3)

        if (self.get_variable('rec_dicom')):
            self.create_dicom()

    def show_tab(self, number):
        main_notebook = self.builder.get_object("main_notebook")
        main_notebook.tab(number, state='normal')

    def disable_tab(self, number):
        main_notebook = self.builder.get_object("main_notebook")
        main_notebook.tab(number, state='disabled')

    def hide_tab(self, number):
        main_notebook = self.builder.get_object("main_notebook")
        main_notebook.tab(number, state='hidden')

    def validate_date(self):
        return True

    def show_calendar(self):
        self.calendar_button['state'] = ACTIVE
        self.calendar.grid(row=1,column=0)

    def on_cell_clicked(self, event):
        self.set_date(event.widget.selection)

    def set_date(self, date):
        self.date_study.configure(state='normal')
        self.date_study.delete(0,END)
        self.date_study.insert(0,date.strftime("%d/%m/%Y"))
        self.date_study.configure(state='readonly')
        self.calendar_button['state'] = NORMAL
        self.calendar.grid_remove()

    def __init__(self):

        self.builder = builder = pygubu.Builder()
        builder.add_from_file('resources/gui.ui')

        self.root = builder.get_object('mainwindow')

        self.calendar = builder.get_object('calendar_input')
        self.calendar.grid_remove()

        self.calendar_button = builder.get_object('calendar_button')
        self.date_study = builder.get_object('date_study')

        # Selekt typu w sinogramie
        self.select_sinogram = builder.get_object('select_sinogram')
        self.select_sinogram.configure(values=["Z animacją", "Gotowy rezultat"])
        self.select_sinogram.set("Z animacją")

        # Selekt typu w rekonstrukcji
        self.select_rec_result = builder.get_object('select_rec_result')
        self.select_rec_result.configure(values=["Gotowy rezultat bez splotu", "Gotowy rezultat ze splotem", "Animacja bez splotu"])
        self.select_rec_result.set("Gotowy rezultat bez splotu")

        self.panel = builder.get_object("image_frame")

        self.f2 = builder.get_object("sinogram_result")

        self.f3 = builder.get_object("rec_result")

        self.f4 = builder.get_object("dicom_result")

        self.slider_alpha = builder.get_object("slider_alpha")
        self.slider_detectors = builder.get_object("slider_detectors")
        self.slider_range = builder.get_object("slider_range")

        self.typ = builder.get_object("select_sinogram")
        self.typ.configure(values=["ANIMACJA","GOTOWY REZULTAT"])
        self.typ.set("ANIMACJA")

        self.disable_tab(1)
        self.disable_tab(2)
        self.hide_tab(3)

        self._filename = 'images/marchewka.jpg'
        self._show_basic_image()

        self.text_comments = builder.get_object("text_comments")

        self.set_date(datetime.now())
        self.calendar.select_day(datetime.now().day)

        builder.connect_callbacks(self)

        self.root.mainloop()

def mse(orig, final):
    err = np.sum((orig.astype("float64") - final.astype("float64"))**2)
    err /= float(orig.shape[0] * orig.shape[1])
    return math.sqrt(err)

if __name__ == '__main__':
    gui = GUI()



