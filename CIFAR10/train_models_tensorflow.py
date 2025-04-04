import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import os
import time

# 定义 VGG11 模型
def VGG11(num_classes=10):
    model = models.Sequential()
    # Conv Block 1
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Conv Block 2
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Conv Block 3
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Conv Block 4
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Conv Block 5
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))

    # 全连接层
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# 加载 CIFAR-10 数据集
print("Loading CIFAR-10 dataset...")
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# 数据预处理 - 归一化和标签的 one-hot 编码
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

# 构建模型
model = VGG11(num_classes=10)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 打印模型结构
model.summary()

# 设置训练参数
batch_size = 128
num_epochs = 50
model_path = "./models"
if not os.path.exists(model_path):
    os.makedirs(model_path)
best_model_file = os.path.join(model_path, "CIFAR10_best_model.h5")

# 回调函数，用于保存最好的模型
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_file, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

# 训练模型
print("Start training...")
start_time = time.time()

history = model.fit(
    X_train, Y_train,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(X_test, Y_test),
    callbacks=[checkpoint],
    verbose=2
)

end_time = time.time()
elapsed_time = end_time - start_time
mins, secs = divmod(elapsed_time, 60)
print(f"Training finished! Total time: {int(mins)} minutes and {int(secs)} seconds.")

# 评估模型
print("Evaluating the model on test set...")
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
