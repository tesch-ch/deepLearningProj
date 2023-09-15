# Standard library imports
import time

# Third-party imports
from transformers import (DetrImageProcessor, DetrForObjectDetection,
                          CLIPProcessor, CLIPModel)
import torch


class ObjectDetector:
    
    def __init__(self):
        # Model selection
        OD_MODEL_PATH = "facebook/detr-resnet-50"
        ZS_MODEL_PATH = "openai/clip-vit-large-patch14"

        # Zero-shot texts
        self.ZS_BED = ["tidy bed", "messy bed"]
        self.ZS_PERSON = ['person wearing black shirt',
                          'person wearing grey shirt',
                          'person wearing white shirt']

        # Set up object detector
        print(f"\nInitializing {OD_MODEL_PATH}")
        self.od_processor = DetrImageProcessor.from_pretrained(OD_MODEL_PATH)
        self.od_model = DetrForObjectDetection.from_pretrained(OD_MODEL_PATH)

        # Set up zero-shot classifier
        print(f"\nInitializing {ZS_MODEL_PATH}")
        self.zs_processor = CLIPProcessor.from_pretrained(ZS_MODEL_PATH)
        self.zs_model = CLIPModel.from_pretrained(ZS_MODEL_PATH)

        # Try to move models to GPU
        self.gpu_available = torch.cuda.is_available()

        if self.gpu_available:
            self.device = torch.device("cuda")
            try:
                self.od_model.to(self.device)
                self.zs_model.to(self.device)
                print("\nMoved models to GPU.")
            except Exception as e:
                print(f"\nGPU available but unable to use. Using CPU instead."
                      f"\nError: {e}")
        else:
            print("\nGPU not available. Using CPU instead.")


    def detect(self, image, od_threshold=0.9):
        """image is a PIL image object, threshold is a float between 0 and 1
        that determines the minimum confidence score for a detection to be
        returned.
        """
        # Run object detection
        od_results = self.object_detection(image, threshold=od_threshold)

        # Get a list of detected labels
        labels = [d["label"] for d in od_results["detections"]]

        # Run zero-shot classification conditional on detected labels
        # TODO: makes no sense on multiple occurrences of the same label,
        # but this is a demo so... Might want to crop image to bounding box etc
        # In order to keep inference time low, we only run zero-shot
        # classification once, hence if else...

        if "person" in labels:
            zs_results = self.zero_shot_classification(
                texts=self.ZS_PERSON, image=image)
            od_results.update(zs_results)
        
        elif "bed" in labels:
            zs_results = self.zero_shot_classification(
                texts=self.ZS_BED, image=image)
            od_results.update(zs_results)
        
        return od_results


    def object_detection(self, image, threshold=0.9):
        """image is a PIL image object, threshold is a float between 0 and 1
        that determines the minimum confidence score for a detection to be
        returned.
        """
        inputs = self.od_processor(images=image, return_tensors="pt")
        
        if self.gpu_available:
            # move inputs to GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        start = time.perf_counter()
        outputs = self.od_model(**inputs)
        inference_time = time.perf_counter() - start

        # Post-processing the results
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.od_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold)[0]
        
        detections = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]):

            # Convert label to string
            label = self.od_model.config.id2label[label.item()]
            
            # Get confidence score
            confidence = round(score.item(), 2)

            # List of box coordinates [x_min, y_min, x_max, y_max]
            box = [int(i) for i in box.tolist()]

            detections.append({
                "label": label,
                "confidence": confidence,
                "box": box
            })
        
        return {"od_time_s": inference_time, "detections": detections}


    def zero_shot_classification(self, texts, image):
        
        inputs = self.zs_processor(text=texts, images=image,
                                   return_tensors="pt", padding=True)
        if self.gpu_available:
            # move inputs to GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        start = time.perf_counter()
        outputs = self.zs_model(**inputs)
        inference_time = time.perf_counter() - start

        # Post-processing the results

        # this is the image-text similarity score
        logits_per_image = outputs.logits_per_image 

        # we can take the softmax to get the label probabilities
        probs = logits_per_image.softmax(dim=1)
        
        # get the text (label) with the highest probability and its probability
        label = texts[probs.argmax()]
        label_prob = float(probs[0][probs.argmax()])
        
        return {"zs_time_s": inference_time, "zs_label": label,
                "zs_label_prob": label_prob}
    

if __name__ == "__main__":
    
    from PIL import Image
    import requests

    # Initialize object detector
    detector = ObjectDetector()

    # Load image from URL
    # It's that cats on a couch with remotes image...
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # # Load image from file
    # input_image = Image.open("mirror_person_grey.jpg")
    # # Define the desired width
    # desired_width = 640
    # width, height = input_image.size
    # aspect_ratio = width / height
    # desired_height = int(desired_width / aspect_ratio)
    # # Resize the image to the desired width and height
    # image = input_image.resize((desired_width, desired_height), Image.ANTIALIAS)

    print('\n\nStart detecting.\n')

    # Run object detection
    od_results = detector.object_detection(image)
    print(od_results, '\n\n')

    # Run zero-shot classification
    zs_results = detector.zero_shot_classification(
        texts=["cat on a couch", "dogs on a couch", "cats on a table"],
        image=image)
    print(zs_results, '\n\n')

    # Run object detection and zero-shot classification
    results = detector.detect(image)
    print(results, '\n\n')


