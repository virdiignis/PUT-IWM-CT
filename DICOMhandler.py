import numpy as np
import pydicom
from matplotlib.image import imread
from pydicom.data import get_testdata_files
from pydicom.dataset import Dataset, FileDataset
from datetime import datetime
import matplotlib.pyplot as plt


class DICOMhandler:
    class Patient:
        def __init__(self, id, name, birthdate, sex):
            self.id = id
            self.name = name
            self.birthdate = birthdate
            self.sex = sex

    class DICOMexamination:
        def __init__(self, filename: str):
            self._dataset = pydicom.dcmread(filename)
            d = self._dataset
            self.patient = DICOMhandler.Patient(d.PatientID, str(d.PatientName), d.PatientBirthDate, d.PatientSex)
            self.date = datetime.strptime(d.StudyDate + d.StudyTime, '%Y%m%d%H%M%S')
            try:
                self.comment = str(d.data_element('ImageComments'))
            except KeyError:
                self.comment = ""
            self.bitmap = d.pixel_array
            if len(self.bitmap.shape) > 2:
                self.bitmap = self.bitmap[:, :, 0]

    def load(self, filename):
        dicom_exam = self.DICOMexamination(filename)
        return dicom_exam

    @staticmethod
    def new(filename, bitmap, metadata):
        tfn = get_testdata_files('CT_small.dcm')[0]
        ds = pydicom.dcmread(tfn)

        ds.Rows, ds.Columns = bitmap.shape
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.SamplesPerPixel = 1
        ds.PixelRepresentation = 0
        ds.PhotometricInterpretation = 'MONOCHROME2'

        pixel_data = bitmap
        pixel_data -= np.min(pixel_data)
        pixel_data *= 65536 / np.max(pixel_data)
        pixel_data = pixel_data.astype(np.uint16)
        pixel_data = pixel_data.tobytes()
        ds.PixelData = pixel_data

        ds.AccessionNumber = '1'
        ds.SeriesNumber = '1'
        ds.PatientOrientation = 'horizontal'
        ds.ReferringPhysicianName = "House M.D."
        ds.StudyID = metadata['patient'].id
        ds.PatientID = metadata['patient'].id
        ds.PatientName = metadata['patient'].name
        ds.PatientBirthDate = metadata['patient'].birthdate
        ds.PatientSex = metadata['patient'].sex
        ds.StudyDate = metadata['date'].strftime('%Y%m%d')
        ds.StudyTime = metadata['date'].strftime('%H%M%S')
        ds.ImageComments = metadata['comments']

        # # Set the transfer syntax
        # ds.is_little_endian = True
        # ds.is_implicit_VR = True

        # Set creation date/time
        dt = datetime.now()
        ds.ContentDate = dt.strftime('%Y%m%d')
        ds.InstanceCreationDate = dt.strftime('%Y%m%d%H%M%S')
        timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
        ds.ContentTime = timeStr
        ds.save_as(filename)


if __name__ == '__main__':
    # h = DICOMhandler()
    # # bitmap = imread('images/Kropka.jpg')[:, :, 0]
    #
    # r = Radon("images/CT_ScoutView-large.jpg", np.pi / 360, 360, 270 * np.pi / 180)
    # r.sinogram()
    # # r.show_sinogram()
    # r.reconstruct(filter=False)
    # # r.show_reconstruction()
    #
    # bitmap = r._reconstructed_bitmap
    #
    # p = DICOMhandler.Patient('1000', "Elon Musk", "19930409", 'M')
    # date = datetime.now()
    # comment = "hehehehehe"
    #
    # metadata = {
    #     "patient": p,
    #     'date': date,
    #     'comments': comment
    # }
    #
    # h.new("test.dcm", bitmap, metadata)
    # print("Ok")
    # o = h.load('/home/prance/PycharmProjects/IwM/CT/test.dcm')
    # plt.imshow(o.bitmap, cmap='gray')
    # plt.show()
    pass
