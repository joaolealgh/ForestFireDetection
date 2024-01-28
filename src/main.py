import os
import argparse

import tensorflow as tf

from train_finetune import train
from forest_fire_model import load_custom_model, mobile_net_transfer_learning, mobile_net_transfer_learning, forest_fire_model, predict_fire, get_base_model_information
from video_labeling import video_to_frames


physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

# Necessary for WSL2 and OpenCV to work
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

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