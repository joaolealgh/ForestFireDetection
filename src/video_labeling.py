from tensorflow.keras.utils import load_img, img_to_array
import time
import os
import numpy as np
import cv2

from forest_fire_model import predict_fire

def video_to_frames(video_path, output_path, model, frame_rate, video_width, video_height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter('video.avi', fourcc, frame_rate, (video_width, video_height))

    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(video_path)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    frame_count = 0

    prediction_frequency = 12 # predict if there is a fire every 12th frame
    img = None

    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Write the results back to output location.
        img_name = f"temp_frame_{frame_count}.png"
        img_path = output_path + img_name

        cv2.imwrite(img_path, frame)

        if frame_count % prediction_frequency == 0:
            img = load_img(img_path, target_size=(224, 224))
            img = img_to_array(img)
            img = np.array([img])
            prediction = predict_fire(model, img)[0][0]

            # load image for writing the video
            img = cv2.imread(img_path)

            # convert image to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # setup text
            font = cv2.FONT_HERSHEY_SIMPLEX
            textsize = cv2.getTextSize(prediction, font, 1, 2)[0]

            textX = (img.shape[1] - textsize[0]) // 2
            textY = (img.shape[0] + textsize[1]) // 2

            rectangle_bgr = (0, 0, 0)

            # make the coordinates of the box
            box_coords = ((textX, textY), (textX + textsize[0], textY - textsize[1]))
            cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)

            # add text centered on image
            cv2.putText(img, prediction, (textX, textY), font, 1, (255, 255, 255), 2)

        else:
            # load image for writing the video
            img = cv2.imread(img_path)

            # convert image to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # setup text
            font = cv2.FONT_HERSHEY_SIMPLEX
            textsize = cv2.getTextSize(prediction, font, 1, 2)[0]

            textX = (img.shape[1] - textsize[0]) // 2
            textY = (img.shape[0] + textsize[1]) // 2

            rectangle_bgr = (0, 0, 0)

            # make the coordinates of the box
            box_coords = ((textX, textY), (textX + textsize[0], textY - textsize[1]))
            cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)

            # add text centered on image
            cv2.putText(img, prediction, (textX, textY), font, 1, (255, 255, 255), 2)

        video.write(img)
        
        frame_count += 1
        # If there are no more frames left
        if (frame_count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % frame_count)
            print ("It took %d seconds for conversion." % (time_end-time_start))
            break
    
    video.release()
