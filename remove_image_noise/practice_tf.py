import os
import cv2
import keras.optimizers as optimizers
from keras.layers import Input, Conv2D
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K


def load_images(inputpath, imagesize, type_color):
    imglist = []
    
    exclude_prefixes = ('__', '.')
    for root, dirs, files in os.walk(inputpath):
        dirs[:] = [dir for dir in dirs if not dir.startswith(exclude_prefixes)]
        files[:] = [file for file in files if not file.startswith(exclude_prefixes)]
        
        for fn in sorted(files):
            bn, ext = os.path.splitext(fn)
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
        print('save_image\n\n')


image_train, image_train_filenames = load_images("./train/", 256, 'Gray')
image_test, image_test_filenames = load_images("./test/", 256, 'Gray')

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


# リスト7.7
CHANNEL = 1


def network_denoisetestnet():
    input_img = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNEL))
    
    x = Conv2D(64, kernel_size=3, activation='relu', padding='same')(input_img)
    x = Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv2D(1, kernel_size=3, activation='linear', padding='same')(x)
    
    model = Model(input_img, x)
    
    return model


model = network_denoisetestnet()
print(model.summary())


# リスト7.8
def psnr(y_true, y_pred):
    return -10 * K.log(K.mean(K.flatten((y_true - y_pred)) ** 2)) / np.log(10)


# traningの設定
adam = optimizers.Adam(lr=1e-3)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[psnr])

# traing実行
training = model.fit(imagenoise_train, image_train, epochs=50, batch_size=10, shuffle=True, validation_data=(imagenoise_test, image_test), verbose=1)


# リスト7.9
# 学習履歴グラフ表示
# val_psnr&val_loss:検証用データセットに対する値, psnr&loss:学習用データセットに対する値
def plot_history(history):
    plt.plot(history.history['psnr'])
    plt.plot(history.history['val_psnr'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['psnr', 'val_psnr'], loc='lower right')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()


plot_history(training)


# リスト7.10
# networkの性能を確認するためにノイズ画像からノイズを取り除いた画像を出力
results = model.predict(imagenoise_test, verbose=1)

results *= 255.0
save_images("./result/", imagenoise_test_filenames, results)
