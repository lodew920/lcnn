import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import h5py
import glob
import time
from random import shuffle
from collections import Counter
import os
from pathlib import PurePath
from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD, Adam
from keras.models import load_model
from mpl_toolkits.axes_grid1 import AxesGrid
map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson',
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel',
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson',
        11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak',
        14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}

img_width = 42
img_height = 42
num_classes = len(map_characters) # 要辨識的角色種類
pictures_per_class = 1000 # 每個角色會有接近1000張訓練圖像
test_size = 0.15
imgsPath = ".\\train"

def load_pictures():
    pics=[]
    labels=[]
    for k, v in map_characters.items():
        pictures = [k for k in glob.glob(imgsPath+"/"+v+"/*")]
        print(v+":"+str(len(pictures)))
        for i, picture in enumerate(pictures):
            tmp_img = cv2.imread(picture)
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
            tmp_img = cv2.resize(tmp_img, (img_height, img_width))
            pics.append(tmp_img)
            labels.append(k)
    return np.array(pics), np.array(labels)

def get_dataset(save=False, load=False):
    if load==True:
        h5f = h5py.File("dataset.h5","r")
        X_train = h5f['X_train'][:]
        X_test = h5py["X_test"][:]
        h5f.close()

        h5f = h5py.File("labels.h5","r")
        y_train = h5f['y_train'][:]
        y_test = h5f["y_test"][:]
        h5f.close()
    else:
        X,y = load_pictures()
        y = keras.utils.to_categorical(y, num_classes)
        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=test_size)
        if save==True:
            h5f = h5py.File("dataset.h5", "w")
            h5f.create_dataset("X_train", data=X_train)
            h5f.create_dataset("X_test", data=X_test)
            h5f.close()

            h5f = h5py.File("labels.h5", "w")
            h5f.create_dataset("X_train", data=y_train)
            h5f.create_dataset("X_test", data=y_test)
            h5f.close()
    X_train = X_train.astype("float32")/255.
    X_test = X_test.astype("float32")/255.
    print("Train", X_train.shape, y_train.shape)
    print("Test", X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test
# -----第一步：加载数据集-----
X_train, X_test, y_train, y_test = get_dataset(save=True, load=False)

#建立一个6层的网络模型
def create_model_six_conv(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation="softmax"))

    return model

# -----第二步：建立模型-----
# model = create_model_six_conv((img_height, img_width, 3))
# model.summary()

# 定义损失函数，可以使用交叉熵损失
# 定义优化器，可以用SGD
# 定义模型评估，可以用准确率
# lr = 0.01
# sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

#训练模型

def lr_schedule(epoch):
    return lr*(0.1**int(epoch/2))
batch_size=32
epochs=1
# -----第三步：训练模型-----
# history = model.fit(X_train, y_train,
#                     batch_size=batch_size, epochs=epochs,
#                     validation_data=(X_test, y_test),
#                     shuffle=True,
#                     callbacks=[LearningRateScheduler(lr_schedule), ModelCheckpoint("model.h5", save_best_only=True)]
#                     )

# # 可视化
#
# def plot_train_history(history, train_metrics, val_metrics):
#     plt.plot(history.history.get(train_metrics), '-o')
#     plt.plot(history.history.get(val_metrics), '-o')
#     plt.ylabel(train_metrics)
#     plt.xlabel('Epochs')
#     plt.legend(['train', 'validation'])
#
#
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plot_train_history(history, 'loss', 'val_loss')
#
# plt.subplot(1, 2, 2)
# plot_train_history(history, 'acc', 'val_acc')
#
# plt.show()
# -----第四步：模型评估-----
def load_test_set(path):
    pics, labels = [], []
    reverse_dict = {v: k for k, v in map_characters.items()}
    for pic in glob.glob(path+"*.*"):
        char_name = "_".join(os.path.basename(pic).split("_")[:-1])
        if char_name in reverse_dict:
            temp = cv2.imread(pic)
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            temp = cv2.resize(temp, (img_height, img_width)).astype("float32")
            pics.append(temp)
            labels.append(reverse_dict[char_name])
    X_test = np.array(pics)
    y_test = np.array(labels)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return X_test, y_test

imgsPath = ".\\test\\"
X_valtest, y_valtest = load_test_set(imgsPath)

# model = load_model("model.h5")
# y_pred = model.predict_classes(X_valtest)
# accuracy = np.sum(y_pred==np.argmax(y_valtest, axis=1))/np.size(y_pred)
# print("Test accuracy = {}".format(accuracy))

#数据增强
# datagen = ImageDataGenerator(
#     featurewise_center=False,
#     samplewise_center=False,
#     featurewise_std_normalization=False,
#     samplewise_std_normalization=False,
#     zca_whitening=False,
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True,
#     vertical_flip=False)
#
# datagen.fit(X_train)
# filepath = "model-dtaug.h5"
# checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=0, save_best_only=True, mode="max")
# def lr_schedule(epoch):
#     return lr*(0.1**int(epoch/2))
# model = create_model_six_conv((img_height, img_width, 3))
# lr = 0.1
# sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])
# callbacks_list=[LearningRateScheduler(lr_schedule), checkpoint]
# history = model.fit_generator(
#     datagen.flow(X_train, y_train, batch_size=batch_size),
#     steps_per_epoch=X_train.shape[0],
#     epochs=epochs,
#     validation_data=(X_test, y_test),
#     callbacks=callbacks_list
# )
# model = load_model("model-dtaug.h5")
# y_pred = model.predict_classes(X_valtest)
# acc = np.sum(y_pred==np.argmax(y_valtest, axis=1))/np.size(y_pred)
# print("test accuracy = {}".format(acc))



