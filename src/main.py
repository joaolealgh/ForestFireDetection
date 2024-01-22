import os
import argparse

import tensorflow as tf
from matplotlib import pyplot as plt

from dataset import download_dataset
from dataset import display_dataset_images
from dataset import prepare_datasets
from forest_fire_model import load_custom_model, mobile_net_transfer_learning, mobile_net_transfer_learning, forest_fire_model, predict_fire, get_base_model_information
from video_labeling import video_to_frames


physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

# Necessary for WSL2 and OpenCV to work
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

def train(transfer_model, download_ds, dataset_path, display_dataset, model_path):
    if download_ds:
        download_dataset(dataset_path)

    if display_dataset:
        display_dataset_images(dataset_path)

    TRAIN_DATA_DIR = dataset_path + '/Data/Train_Data/' # take into consideration the path: dataset_path
    TEST_DATA_DIR = dataset_path + '/Data/Test_Data/' # take into consideration the path: dataset_path

    model_data = get_base_model_information(transfer_model)

    IMG_HEIGHT = model_data['IMG_HEIGHT']
    IMG_WIDTH = model_data['IMG_WIDTH']
    BATCH_SIZE = 8
    SEED = 123
    CHANNELS = 3
    INITIAL_EPOCHS = 5

    train_ds, val_ds, test_ds = prepare_datasets(TRAIN_DATA_DIR, TEST_DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, SEED)

    base_model, preprocess_input = mobile_net_transfer_learning(IMG_HEIGHT, IMG_WIDTH, CHANNELS, train_ds)
    model = forest_fire_model(preprocess_input, base_model, train_ds, val_ds, test_ds, epochs=INITIAL_EPOCHS, model_path=model_path)

    # Positive values predict class 1 -> Non Fire
    # Negative values predict class 0 -> Fire
    # y_hat = model.predict(test_ds)
    # y_hat[y_hat > 0] = 1
    # y_hat[y_hat < 0] = 0

    # TODO: Encapsulate in a function
    fig, ax = plt.subplots(nrows=8, ncols=6, figsize=(20, 40))
    j = 0
    for batch_x, batch_y in test_ds:
        if j==6:
            break
        predicted_batch = predict_fire(model, batch_x)
        # print(predicted_batch)
        for i, x in enumerate(batch_x):
            ax[i, j].imshow(x.numpy().astype("uint8"))
            ax[i, j].axis("off")
            predicted_label = predicted_batch[i][0]
            ax[i, j].set_title(f'{predicted_label}')
        j+=1
    plt.show()


def main(action, base_model, download_ds, dataset_path, display_dataset, video_path, video_annotated, video_width, video_height, model_path):
    if action == 'TRAIN':
        train(base_model, download_ds, dataset_path, display_dataset, model_path)

    elif action == 'INFERENCE':
        # TODO: Allow user to choose the framerate and the frequency of frames to be predicted
        frame_rate = 24

        model = load_custom_model(path=model_path)
        video_to_frames(video_path=video_path, output_path=video_annotated, model=model, frame_rate=frame_rate, video_width=video_width, video_height=video_height)


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument('--action',
                        '-a',
                        choices=['TRAIN', 'INFERENCE'], 
                        required=True,
                        help='TRAIN or INFERENCE')

    parser.add_argument('--base-model',
                        '-b',
                        choices=['MobileNetV2'],
                        default='MobileNetV2',
                        help='Base model to use (MobileNetV2, etc)')

    parser.add_argument('--download-dataset',
                        default=False,
                        type=bool,
                        help='Download dataset from original source')

    parser.add_argument('--dataset-path',
                        default='../datasets',
                        help='Path to download the dataset to')

    parser.add_argument('--display-dataset', 
                        default=False,
                        type=bool,
                        help='Display a subset of images from dataset')

    parser.add_argument('--video-path',
                        help='Video path to annotate')

    parser.add_argument('--video-annotated', 
                        default='outputs/',
                        help='Annotated video output path')
    
    parser.add_argument('--video-width',
                        type=int,
                        help='Input video width dimension')
    
    parser.add_argument('--video-height',
                        type=int,
                        help='Input video height dimension')

    parser.add_argument('--model-path', 
                        default='model/feature_extraction_model.h5', 
                        help='Location to save the model if action is TRAIN or location to load the model from if action is INFERENCE')

    args = parser.parse_args()
    
    return args

if __name__ =='__main__':
    args = run()
    
    action = args.action
    base_model = args.base_model
    download_ds = args.download_dataset
    dataset_path = args.dataset_path
    display_dataset = args.display_dataset
    video_path = args.video_path
    video_annotated = args.video_annotated
    video_width = args.video_width
    video_height = args.video_height
    model_path = args.model_path

    main(action=action, 
         base_model=base_model, 
         download_ds=download_ds, 
         dataset_path=dataset_path, 
         display_dataset=display_dataset, 
         video_path=video_path, 
         video_annotated=video_annotated, 
         video_width=video_width,
         video_height=video_height,
         model_path=model_path)