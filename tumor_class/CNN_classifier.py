from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras import callbacks
from sklearn.metrics import accuracy_score
import numpy as np
from PIL import Image, ImageChops, ImageOps
from os import listdir, mkdir
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def process_images(directories):
    mkdir('processed_images')
    for files in ['Training/', 'Testing/']:
        mkdir('processed_images/'+files)
        for file in directories:
            mkdir('processed_images/'+files+file)
            for filename in listdir(files+file):
                image = Image.open(files+file+'/'+filename)
                gs_image = image.convert(mode='L')
                gs_image = trim(gs_image)
                gs_image = gs_image.resize((100, 100))
                gs_image.save('processed_images/'+files+file+'/processed_'+filename)
                # #gs_image_flip = ImageOps.flip(gs_image)
                # for angle in [0, 90, 180, 270]:
                #     gs_image.rotate(angle).save('processed_images/'+file+'/processed_'+str(angle)+filename)
                #     #gs_image_flip.rotate(angle).save('processed_images/' + file + '/processed_flip'+str(angle)+filename)


def build_training_set(directories, dir):
    training_images, testing_images = [], []
    training_labels, testing_labels = [], []
    outputs = {'glioma_tumor': [1, 0, 0, 0], 'meningioma_tumor': [0, 1, 0, 0], 'no_tumor': [0, 0, 1, 0], 'pituitary_tumor': [0, 0, 0, 1]}
    for file in directories:
        dir_file = 'processed_images/'+'Training/'+file
        for filename in listdir(dir_file):
            image = Image.open(dir_file+'/'+filename)
            training_images.append(np.asarray(image))
            training_labels.append(outputs[file])
    training_images = np.array(training_images, dtype='float32')
    training_images = np.expand_dims(training_images, axis=-1)
    training_labels = np.array(training_labels, dtype='float32')
    training = np.random.permutation(training_labels.shape[0])
    for file in directories:
        dir_file = 'processed_images/'+'Testing/'+file
        for filename in listdir(dir_file):
            image = Image.open(dir_file+'/'+filename)
            testing_images.append(np.asarray(image))
            testing_labels.append(outputs[file])
    testing_images = np.array(testing_images, dtype='float32')
    testing_images = np.expand_dims(testing_images, axis=-1)
    testing_labels = np.array(testing_labels, dtype='float32')
    testing = np.random.permutation(testing_labels.shape[0])
    return training_images[training], training_labels[training], testing_images[testing], testing_labels[testing]


def build_model(outputs, shape):
    model = Sequential()
    #model.add(Conv2D(32, (7, 7), activation='relu', input_shape=shape))
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(.2))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(.2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(outputs, activation='softmax'))
    return model


def cm(y_true, y_pred):
    cm2 = 0
    b = np.zeros_like(y_pred)
    b[np.arange(len(predictions)), predictions.argmax(1)] = 1
    for i in range(len(y_true)):
        ind = np.argpartition(y_pred[i], -2)[-2:]
        x = np.array([1. if i == ind[1] else 0. for i in range(4)])
        y = np.array([1. if i == ind[0] else 0. for i in range(4)])
        if str(x) == str(y_true[i]) or str(y) == str(y_true[i]):
            cm2 += 1
    cm = np.zeros((4, 4))
    for i in range(len(y_true)):
        cm[y_true[i].tolist().index(1)][y_pred[i].tolist().index(1)] += 1
    cm = cm/cm.sum(axis=0)
    df_cm = pd.DataFrame(cm, index=[i for i in ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']], columns=[i for i in ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']])
    plt.figure(figsize=(4, 4))
    sn.heatmap(df_cm, annot=True)
    plt.title(f'Accuracy within two first guesses {round(cm2/len(y_true)*100, 4)}%')
    plt.show()


#process_images(['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'])
images, label, t_images, t_label = build_training_set(['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'], 'processed_images')
model = build_model(4, images.shape[1:])
opt = Adam(learning_rate=1e-3, decay=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
earlystopping = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=20, restore_best_weights=True)
model.fit(images, label, epochs=70, verbose=1, callbacks=[earlystopping], validation_split=0.1)

predictions = model.predict(t_images)
b = np.zeros_like(predictions)
b[np.arange(len(predictions)), predictions.argmax(1)] = 1
print('Accuracy:', accuracy_score(t_label, b))

cm(t_label, b)

for i in range(30):
    outputs = {'[1. 0. 0. 0.]': 'glioma_tumor', '[0. 1. 0. 0.]': 'meningioma_tumor', '[0. 0. 1. 0.]': 'no_tumor', '[0. 0. 0. 1.]': 'pituitary_tumor'}
    plt.imshow(t_images[i])
    ind = np.argpartition(predictions[i], -2)[-2:]
    plt.title(f'Predicted: {outputs[str(np.array([1. if i == ind[1] else 0. for i in range(4)]))]}({round(predictions[i][ind[1]]*100, 4)}%) {outputs[str(np.array([1. if i == ind[0] else 0. for i in range(4)]))]}({round(predictions[i][ind[0]]*100, 4)}%) \n Actual: {outputs[str(t_label[i])]}')
    plt.show()