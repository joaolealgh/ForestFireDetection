import os
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from dotenv import load_dotenv
from zipfile import ZipFile
import cv2

load_dotenv()

def download_dataset(dataset_path):
    # 'username' and 'key' are in a .env file
    os.environ['KAGGLE_USERNAME']=os.getenv('username')
    os.environ['KAGGLE_KEY']=os.getenv('key')

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    # kaggle datasets download -d mohnishsaiprasad/forest-fire-images
    os.system('kaggle datasets download -d mohnishsaiprasad/forest-fire-images')

    with ZipFile('forest-fire-images.zip', 'r') as f:
        f.extractall('../datasets/')

    os.remove('forest-fire-images.zip')

def display_dataset_images(dataset_path):
    fire_train_folder = dataset_path + '/Data/Train_Data/Fire/'
    non_fire_train_folder = dataset_path + '/Data/Train_Data/Non_Fire/'

    num_images_load = 3

    fire_train_images = []
    for filename in os.listdir(fire_train_folder):
        img_path = os.path.join(fire_train_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fire_train_images.append(img)
        else:
            # Remove image from folder
            print(f'Removing image: {img_path}')
            os.remove(img_path)

    non_fire_train_images = []
    for filename in os.listdir(non_fire_train_folder):
        img_path = os.path.join(non_fire_train_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            non_fire_train_images.append(img)
        else:
            # Remove image from folder
            print(f'Removing image: {img_path}')
            os.remove(img_path)

    print(f'Number of Fire images: {len(fire_train_images)}\nNumber of Non Fire Images: {len(non_fire_train_images)}')

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))

    for i in range(num_images_load):
        # View Fire Images
        fire_image = fire_train_images[i]
        ax[0, i].imshow(fire_image)
        ax[0, i].axis("off")

        # View Non Fire Images
        non_fire_image = non_fire_train_images[i]
        ax[1, i].imshow(non_fire_image)
        ax[1, i].axis("off")

    plt.show()

    # View images dimension

    for i in range(num_images_load):
        (h, w, c) = fire_image.shape[:3]
        print(f'Fire image shape: {h} {w} {c}')

        (h, w, c) = non_fire_image.shape[:3]
        print(f'Non fire image shape:{h} {w} {c}')


def prepare_datasets(TRAIN_DATA_DIR, TEST_DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, SEED):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=TRAIN_DATA_DIR,
        validation_split=0.2,
        subset="training",
        shuffle=True,
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=TRAIN_DATA_DIR,
        validation_split=0.2,
        subset="validation",
        shuffle=True,
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=TEST_DATA_DIR,
        validation_split=None,
        shuffle=False,
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)
    

    return train_ds, val_ds, test_ds