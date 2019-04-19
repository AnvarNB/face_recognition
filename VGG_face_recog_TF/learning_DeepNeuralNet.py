from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import ZeroPadding2D, Convolution2D, Dense, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten

# у себя на компе создаете три каталога (в том же месте, где этот файл обучения нейронки). 
# В них должны находится изображения лиц, 70% - train, 15% - val, 15% - test.

train_dir = 'train'
# каталог с данными для проверки
val_dir = 'val'
# каталог с данными для тестирования
test_dir = 'test'
# размеры изображения
img_width, img_height = 224, 224
# размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# количество эпох
epochs = 1
# размер мини-выборки
batch_size = 100
# количество изображений для обучения
nb_train_samples = 8775
# количество изображений для проверки
nb_validation_samples = 949
# количество изображений для тестирования
nb_test_samples = 1046

datagen = ImageDataGenerator(rescale=1. / 255)

# первый генератор
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# второй генератор
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# третий генератор
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model = Sequential()

model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(100))   # на выходе 100 классов (категорий)
model.add(Activation('softmax'))

# компилируем модель - задаем параметры для обучения
model.compile(loss="categorical_crossentropy", optimizer='SGD', metrics=["accuracy"])
print(model.summary())

# обучаем модель с использованием генераторов
model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=nb_validation_samples // batch_size)

# оцениваем качество обучения сети на тренировочных данных
scores = model.evaluate_generator(train_generator, nb_train_samples // batch_size)
print("Аккуратность работы на тренировочных данных: %.2f%%" % (scores[1]*100))

# оцениваем качество обучения сети на проверочных данных
scores = model.evaluate_generator(val_generator, nb_validation_samples // batch_size)
print("Аккуратность работы на валидационных данных: %.2f%%" % (scores[1]*100))

# оцениваем качество обучения сети на тестовых данных
scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность работы на тестовых данных: %.2f%%" % (scores[1]*100))

# сохраняем обученную сеть
# генерируем описание модели в формате json
model_json = model.to_json()
json_file = open("faces_NN.json", "w")
# записываем архитектуру сети в файл
json_file.write(model_json)
json_file.close()
# записываем данные о весах в файл
model.save_weights("faces_NN.h5")
print("Сохранение сети завершено")

