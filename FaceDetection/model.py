import cv2
import os
import numpy as np

def detect_and_crop_faces(image_path):
    # Load the input image
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    # Load the pre-trained face detector model
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"
    config_path = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    # Prepare the input image for face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network to detect faces
    net.setInput(blob)
    detections = net.forward()

    # Iterate over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Extract the coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding box coordinates fall within the dimensions of the image
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Calculate the height and width of the detected face
            face_height = endY - startY
            face_width = endX - startX

            # Adjust the cropping region to include the top of the head and bottom of the chest
            top_margin = int(face_height * 0.5)  
            bottom_margin = int(face_height * 1.0)  
            side_margin = int(face_width * 0.5)  

            startY = max(0, startY - top_margin)
            endY = min(h - 1, endY + bottom_margin)
            startX = max(0, startX - side_margin)
            endX = min(w - 1, endX + side_margin)

            # Crop the face region from the image
            face = image[startY:endY, startX:endX]

            # Resize the cropped face to passport size (3.5x4.5 cm)
            resized_face = cv2.resize(face, (354, 472), interpolation=cv2.INTER_AREA)

            # Create an output directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(image_path), "output")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save the cropped face
            output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_face{i}.jpg")
            cv2.imwrite(output_path, resized_face)
            print("Face cropped and saved to", output_path)

def process_images(folder_path):
    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".jpeg"):
            # Detect and crop faces in each image
            image_path = os.path.join(folder_path, file_name)
            detect_and_crop_faces(image_path)

def main():
    # Input folder containing images
    folder_path = input("Enter the path to the folder containing images: ").strip()

    # Process images
    process_images(folder_path)

if __name__ == "__main__":
    main()
