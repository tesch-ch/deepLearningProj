# Standard library imports
import sys
import threading
import io
import time
import requests

# Third-party imports
import cv2
from PIL import Image


class sharedData:
    def __init__(self):
        self.latest_frame = None
        self.latest_result = None
        self.stop = False
        self.lock_frame = threading.Lock()
        self.lock_result = threading.Lock()
    
    def latest_frame_set(self, frame):
        with self.lock_frame:
            self.latest_frame = frame

    def latest_frame_get(self):
        with self.lock_frame:
            return self.latest_frame
        
    def latest_result_set(self, result):
        with self.lock_result:
            self.latest_result = result

    def latest_result_get(self):
        with self.lock_result:
            return self.latest_result
        

# Frame capture thread
def frame_capture_thread(shared_data):
    # Access the webcam (camera index 0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        sys.exit(0)

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        if ret:
            shared_data.latest_frame_set(frame)
        else:
            print("Error: Could not read a frame from the webcam.")

        if shared_data.stop:
            break

    # Release the webcam
    cap.release()


# Frame processing thread
def frame_processing_thread(shared_data):
    while True:
        frame = shared_data.latest_frame_get()
        if frame is not None:
            # Convert the frame to a PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Send the PIL Image directly to the Flask backend
            with io.BytesIO() as buffer:
                pil_image.save(buffer, format="JPEG")
                frame_data = buffer.getvalue()

            # Send the frame directly to the Flask backend
            response = requests.post('http://localhost:5000/classify',
                                     files={'image': ('frame.jpg', frame_data)})
            shared_data.latest_result_set(response.json())

        if shared_data.stop:
            break


# Start the threads
shared_data = sharedData()
threading.Thread(target=frame_capture_thread, args=(shared_data,)).start()
threading.Thread(target=frame_processing_thread, args=(shared_data,)).start()


# Main thread
while True:
    # TODO: Display the latest frame and result only if there is new frame
    frame = shared_data.latest_frame_get()
    if frame is not None:
        cv2.imshow("Webcam Frame", frame)
    
    result = shared_data.latest_result_get()
    if result is not None:

        # draw the bounding box
        for detection in result["detections"]:
            
            label = detection["label"]
            confidence = detection["confidence"]
            box = detection["box"]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                          (0, 255, 0), 1)
            cv2.putText(frame, f"{label} {confidence}",
                        (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)
            
            if label in ["person", "bed"]:
                # put zs_label and zs_label_prob on the frame additionally
                zs_label = result["zs_label"]
                zs_label_prob = result["zs_label_prob"]
                cv2.putText(frame, f"ZS: {zs_label} {zs_label_prob:.2f}",
                            (box[0], box[1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
        
        # Display inference time
        od_time_s = result["od_time_s"]
        # check if zero-shot classification was performed
        if "zs_time_s" in result:
            zs_time_s = result["zs_time_s"]
            cv2.putText(frame, f"OD: {od_time_s:.2f}s, "
                        f"ZS: {zs_time_s:.2f}s",
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1)
        else:
            cv2.putText(frame, f"OD time: {od_time_s:.2f}s",
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1)
    

    # Check for a key press to exit the loop (e.g., press 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        shared_data.stop = True
        print('Exit by q...')
        # Wait for the threads to finish
        # TODO: implement actual synchronization
        time.sleep(1)
        break

# Close the OpenCV window
cv2.destroyAllWindows()

