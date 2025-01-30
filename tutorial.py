import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

import os
save_dir = os.environ.get('SAKURA_ARTIFACT_DIR')

from logging import getLogger, Formatter, INFO, FileHandler
logger = getLogger(__name__)
logger.setLevel(INFO)

# FileHandlerの設定
if save_dir == None:
    fh = FileHandler('your_log.log')
else:
    fh = FileHandler(f'{save_dir}/your_log.log')
fh.setLevel(INFO)
format = Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s')
fh.setFormatter(format)
logger.addHandler(fh)

# データセットの読み込み
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# データの正規化
x_train = x_train / 255.0
x_test = x_test / 255.0

# モデルの構築
model = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(10, activation='softmax')
])

# モデルのコンパイル
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# モデルの学習
logger.info(f'{device_lib.list_local_devices()}')
if tf.config.list_physical_devices('GPU'):
    logger.info('GPU is available')
else:
    logger.info('GPU is not available')

model.fit(x_train, y_train, 
          epochs=10, 
          batch_size=32, 
          validation_data=(x_test, y_test))

# モデルの保存
if save_dir == None:
    model.save('my_model.keras')
    logger.info('Model is saved to my_model.keras')
else:
    model.save(f'{save_dir}/my_model.keras')
    logger.info(f'Model is saved to {save_dir}/my_model.keras')

# 新しい画像の分類
img_path = 'your_image.jpg'  # ここに分類したい画像のパスを指定
img = image.load_img(img_path, target_size=(32, 32))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

predictions = model.predict(x)
predicted_class = np.argmax(predictions)
logger.info(f'Predicted class: {predicted_class}')
