# Część pierwsza CNN - "Pies czy kot?"

from keras.models import Sequential  # inicjalizacja sieci neuronowej
from keras.layers import Convolution2D  # operacja splotu 2D w sieci
from keras.layers import MaxPool2D  # operacja maxpoolingu 2D w sieci
from keras.layers import Flatten  # wyrównanie do kolumny w sieci
from keras.layers import Dense  # przejście do klasycznej ANN

# Inicjalizacja CNN
classifier = Sequential()

# Operacja splotu
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Operacja maxpoolingu
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Operacja flattingu (wyrównania do kolumny)
classifier.add(Flatten())

# Pełne połączenie sieci
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# Kompilacja CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Dopasowanie zdjęć do sieci neuronowej (CNN) (kod pobrany z dokumentacji Kerasa)
from keras_preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),  # takie samo jak input_shape
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

classifier.fit_generator(
    training_set,
    steps_per_epoch=1000,
    epochs=8,
    validation_data=test_set,
    validation_steps=2000)
