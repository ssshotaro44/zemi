import pydicom
import numpy as np
import cv2

chest_DICOM  = pydicom.read_file('./chestimages_DICOM/JPCLN001.dcm')
w = chest_DICOM.Columns
h = chest_DICOM.Rows
D4d = np.zeros((1, h, w, 1))
tmp = np.zeros((h, w))
tmp =  chest_DICOM.pixel_array
tmp2 = np.reshape(tmp, (h, w, 1))
D4d[0] = tmp2

tmp8bit = tmp/16
tmp8bit.astype(np.uint8)
cv2.imwrite('D8b.png', tmp8bit)