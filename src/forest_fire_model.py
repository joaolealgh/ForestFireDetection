import os
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D, RandomFlip, RandomRotation
from matplotlib import pyplot as plt
import numpy as np

def predict_fire(model, batch):
    y_hat = model.predict(batch)
    # y_hat[y_hat > 0] = 1
    # y_hat[y_hat < 0] = 0
    
    # if y_hat > 1:
    #     return 'Non Fire'
    # else:
    #     return 'Fire'

    return np.where(y_hat > 0, 'Non Fire', 'Fire')


def data_augmentation_layer():
    # Add more options for data augmentation
    data_augmentation = Sequential([
        RandomFlip('horizontal'),
        RandomRotation(0.2),
    ])
    
    return data_augmentation


# Convert into a class
class ForestFireModel():
    # Redo the init values in the model
    def __init__(self, base_model):
        self.base_model = base_model
        self.model = None
        self.preprocess_input = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.history = None

    def build(self, preprocess_input):
        self.preprocess_input = preprocess_input
        data_augmentation = data_augmentation_layer()
        global_average_layer = GlobalAveragePooling2D()
        prediction_layer = Dense(1)

        # Build the model
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = self.base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)

        self.model = Model(inputs, outputs)

        self.model.summary()

        print(len(self.model.trainable_variables))

        base_learning_rate = 0.0001
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy')])

        # loss0, accuracy0 = self.model.evaluate(val_ds)

        # print("initial loss: {:.2f}".format(loss0))
        # print("initial accuracy: {:.2f}".format(accuracy0))        


    def plot_model(self):
        tf.keras.utils.plot_model(self.model, show_shapes=True)


    def train(self, train_ds, val_ds, initial_epochs=5):
        checkpoint_path = "training/feature_extraction_model.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)

        self.history = self.model.fit(train_ds,
                                        epochs=initial_epochs,
                                        validation_data=val_ds,
                                        callbacks=[cp_callback])
        
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()


    def save_custom_model(self, path='feature_extraction_model.h5'):
        self.model.save(path)

    # def load_custom_model(self, path=''):
    #     self.model = load_model(path)
        

def forest_fire_model(preprocess_input, base_model, train_ds, val_ds, test_ds, load, path='feature_extraction_model.h5'):
    if load:
        return load_model(path)
    else:
        # Build model and train on the data
        model = ForestFireModel(base_model)
        model.build(preprocess_input)
        pass

    return model


# Convert into a class
def mobile_net_transfer_learning(IMG_HEIGHT, IMG_WIDTH, CHANNELS, train_ds):
    # Transfer learning starts here

    # Pre processing the input for the MobileNetV2 model
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input 

    # Call the Pre trained MobileNetV2 to serve as the base model which will be fine tuned later

    # TODO Mobile Net max values are 224 x 224, therefore, the dataset needs to be resized from 250 x 250 to 224 x 224 (for this net in specific)
    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    base_model = tf.keras.applications.MobileNetV2(input_shape=INPUT_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
    
    base_model.trainable = False

    base_model.summary()

    image_batch, label_batch = next(iter(train_ds))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)


    return base_model, preprocess_input