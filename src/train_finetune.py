import tensorflow as tf
from matplotlib import pyplot as plt

from dataset import download_dataset
from dataset import display_dataset_images
from dataset import prepare_datasets
from forest_fire_model import mobile_net_transfer_learning, mobile_net_transfer_learning, forest_fire_model, predict_fire, get_base_model_information

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
    _, ax = plt.subplots(nrows=8, ncols=6, figsize=(20, 40))
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
