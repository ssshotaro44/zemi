import os
import cv2
import numpy as np


def load_images(inputpath, imagesize, type_color):
    imglist = []
    
    exclude_prefixes = ('__', '.')
    
    for root, dirs, files in os.walk(inputpath):
        dirs[:] = [dir for dir in dirs if not dir.startswith(exclude_prefixes)]
        files[:] = [file for file in files if not file.startswith(exclude_prefixes)]
        
        for fn in sorted(files):
            bn, ext = os.path.splittext(fn)
            if ext not in ['.bmp', 'BMP', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                continue
            
            filename = os.path.join(root, fn)
            
            if type_color == 'Color':
                testimage = cv2.imread(filename, cv2.IMREAD_COLOR)
                height, width = testimage.shape[:2]
                
                testimage = cv2.resize(testimage, (imagesize, imagesize), interpolation=cv2.INTER_AREA)
                testimage = np.asarray(testimage, dtype=np.float64)
                testimage = testimage[:, :, ::-1]  # チャンネルをBGRからRGBに変更
                
            elif type_color == 'Gray':
                testimage = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                height, width = testimage.shape[:2]
                testimage = cv2.resize(testimage, (imagesize, imagesize), interpolation=cv2.INTER_AREA)
                testimage = np.asarray([testimage], dtype=np.float64)
                testimage = np.asarray(testimage, dtype=np.float64).reshape((imagesize, imagesize, 1))
                
            imglist.append(testimage)
        imgsdata = np.asarray(imglist, dtype=np.float32)
        
        return imgsdata, sorted(files)


def generate_noise(imagelist):
    imagelist_out = []
    
    for i in range(0, len(imagelist)):
        image_temp = imagelist[i] + np.random.normal(loc=0.0, scale=50.0, size=imagelist[i].shape)
        image_temp = np.clip(image_temp, 0, 255)  # 8 bitの範囲に納める
        imagelist_out.append(image_temp)
    
    imgsdata = np.asarray(imagelist_out, dtype=np.float32)
    
    return imgsdata


def save_images(savepath, filenamelist, imagelist):
    for i, fn in enumerate(filenamelist):
        filename = os.path.join(savepath, fn)
        testimage = imagelist[i]
        testimage = testimage[:, :, ::-1]  # 色チャンネルをRGBからBGRへ変換
        cv2.imwrite(filename, testimage)


image_train, image_train_filenames = load_images("/train/", 256, 'Gray')
image_test, image_test_filenames = load_images("/test/", 256, 'Gray')

image_train = generate_noise(image_train)
image_test = generate_noise(image_test)

save_images("./train_noise/", image_train_filenames, image_train)
save_images("./test_noise/", image_test_filenames, image_test)

IMAGE_SIZE = 256

# 現画像とノイズ画像読み込み
imagenoise_train, imagenoise_train_filenames = load_images("./train_noise/", IMAGE_SIZE, 'Gray')
image_train, image_train_filenames = load_images("./train/", IMAGE_SIZE, 'Gray')
imagenoise_test, imagenoise_test_filenames = load_images("./test_noise/", IMAGE_SIZE, 'Gray')
image_test, image_test_filenames = load_images("./test/", IMAGE_SIZE, 'Gray')

# 画素値0-1正規化
imagenoise_train /= 255.0
image_train /= 255.0
imagenoise_test /= 255.0
image_test /= 255.0