# Forest Fires Detection


### Motivation
This project aims to identify random fires that can start in forests. The motivation behind this project is based on the dozens of fires that occur every year in Portugal. Through the usage of Deep Learning models and IoT, systems installed in forests could detect fires as soon as they appear and allow for faster responses by the responsible authorities. This could help to diminish the damages that random fires cause to the environment and to the population close to those areas. This is relevant because every year several acres of land are burned, wildlife is lost, and several habitants lose their houses, and in extreme cases, their lives.


### Project Development Structure

1. Obtain and analyze one or more datasets with images that represent forest fires;

2. Develop a deep learning model (computer vision landscape) that is able to classify correctly if a fire is or not present in an image;

3. Develop a technique to process videos, extract the frames, and classify the frames in order to verify if there is a wild fire or not present;

4. The final output is an annotated video, identical to the input video, that identifies if there is a fire.


### Tools

- Tensorflow/Keras
- OpenCV
- NumPy
- Matplotlib

### Resources

- Forest Fire Images Dataset from https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images/data

- Wildfires 101 | National Geographic (https://www.youtube.com/watch?v=5hghT1W33cY)


### Steps to run

- Create a .env file inside the /src folder with your kaggle API 'username' and 'key'
- Run a Makefile command for TRAIN or INFERENCE
- Done!


### In-progress features

- Makefile
- Docker setup/Docker compose setup
- Deployment with kubernetes
- Logging
- Env vars
- Add additional base transfer models such as ResNet, Inception, EfficientNet, NASNetLarge, Xception, etc 
    (https://medium.com/@blant.jesse/transfer-learning-neur-12df2f55b601)

- Option to disable tensorflow warnings
- Implementation using Pytorch/Other machine learning frameworks
- Front end using Django/Flask/Dash to create a simple website to upload videos and detect fire/non fire
- Organize the code files
- Add more examples
- Support more datasets
- Change paths to use the os.path good practices instead of strings
- More...