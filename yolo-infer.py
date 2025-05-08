import cv2
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo12s.pt")  # load the model

# Define path to video file
source = "sample.mp4"  # path to video file

# Open the video
cap = cv2.VideoCapture(source)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process video frames in a loop
while True:
    # Read a frame
    ret, frame = cap.read()
    
    # Break the loop if we've reached the end of the video
    if not ret:
        break
    
    # Run inference on the frame
    results = model(frame, classes=[73]) 
    
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    
    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()