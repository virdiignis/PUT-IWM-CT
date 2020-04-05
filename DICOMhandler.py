import numpy
import pydicom
from matplotlib.image import imread
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
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        file_meta.MediaStorageSOPInstanceUID = "1.2.3"
        file_meta.ImplementationClassUID = "1.2.3.4"
        file_meta.TransferSyntaxUID = "1.2.840.10008.1.2"

        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

        ds.Columns, ds.Rows = bitmap.shape
        ds.BitsAllocated = 8
        ds.PixelData = bitmap.tobytes()
        ds.SamplesPerPixel = 1
        ds.PixelRepresentation = 0
        ds.PhotometricInterpretation = 'MONOCHROME2'

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

        # Set the transfer syntax
        ds.is_little_endian = True
        ds.is_implicit_VR = True

        # Set creation date/time
        dt = datetime.now()
        ds.ContentDate = dt.strftime('%Y%m%d')
        ds.InstanceCreationDate = dt.strftime('%Y%m%d%H%M%S')
        timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
        ds.ContentTime = timeStr
        ds.save_as(filename)


if __name__ == '__main__':
    h = DICOMhandler()
    bitmap = imread('images/Kropka.jpg')[:, :, 0]
    p = DICOMhandler.Patient('1000', "Elon Musk", "19930409", 'M')
    date = datetime.now()
    comment = "hehehehehe"

    metadata = {
        "patient": p,
        'date': date,
        'comments': comment
    }

    h.new("test.dcm", bitmap, metadata)
    print("Ok")
    o = h.load('/home/prance/PycharmProjects/IwM/CT/test.dcm')
    plt.imshow(o.bitmap, cmap='gray')
    plt.show()
    pass
