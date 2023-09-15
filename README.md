# Realtime object detection with conditional zero-shot classification
This project realizes a prototype of a realtime computer vision system. From a multithreaded frontend app, webcam frames are sent to a remote deployed flask backend with gpu resources.  
On the backend, object detection is performed and for specific labels there is a downstream zero-shot classification in order to get an even deeper insight into the detected objects.

## Components

### Frontend app (`app/post_webcam_mt.py`)
- 3 threads, 1 custom class for data sharing with locking mechanism
  - Frame capture thread
    - continuously captures webcam frames
  - Frame processing thread
    - Converts newest webcam frame to PIL image object and posts it to the backend
    - Shares the results of the inference
  - Main thread
    - Display the newest frame
    - Postprocess results and displays the following features:
      - Inference time of the DETR model
      - Inference time of the clip model
      - DETR's Bounding boxes, labels, confidence scores
      - clip's most likely class name, score

### Backend app (`app/be_app.py`)
- Flask app
- Runs on the data science cluster
- Listens on port 5000
- Receives frame as PIL image object
- Runs inference on the models via `ObjectDetector` (see below)

### Object detection module (`app/detection.py`)
- Utilizes Huggingface transformers
- Contains convenience class `ObjectDetector`
  - Loads two models and tries to move these onto a gpu
  - Performs object Detection via
    - DETR (End-to-End Object Detection) model with ResNet-50 backbone
    - https://huggingface.co/facebook/detr-resnet-50
  - Zero-Shot Image Classification via
    - clip-vit-large-patch14
    - https://huggingface.co/openai/clip-vit-large-patch14
    - Is performed dependent on the labels of the previous object detection
    - Describes detected objects more in detail based on hard coded text input
    - So far only implemented for labels `"person"` and `"bed"`

## Performance
Thanks to the multithreaded design, the webcam frames are always displayed without noticeable lag.  
The real time display of the object detection output on the other hand lags behind the displayed webcam frames. This is caused by the inference time of the models and the overhead caused mostly by the data transfer from frontend to backend and vice versa.  
Without gpu resources, object detection is hardly realtime anymore. Inference time on an 8th gen i7 processor took approx 3 s per model (6 s overall).  
Utilizing a v100 gpu on our kubernetes cluster, inference time goes down to approx 50 ms per model. Using this setup allows for a very smooth and actual real time feeling to the app.

## Result
- With gpu support, app has actual realtime qualities
- No crashing so far
- Object detection with chained zero shot classification works
- Successful proof of concept / prototype

## discussion/issues/ideas
- Crop image according to bounding boxes, run zero shot classification only on the cropped part
- Come up with interesting possible labels for the zero shot classification (utilize GPT for creation of possible labels)