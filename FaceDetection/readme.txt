This Python application detects faces in images and crops them to passport size.

Installation
To run the code in Visual Studio Code, follow these steps:

Install Python: Make sure you have Python installed on your system. You can download it from python.org.

Install OpenCV: OpenCV is used for face detection and image manipulation. You can install it via pip:


pip install opencv-python

Download Pre-trained Model: Download the pre-trained face detection model and configuration files:

deploy.prototxt
res10_300x300_ssd_iter_140000.caffemodel
Place these files in the same directory as your Python script.




Usage
Clone Repository: Clone or download this repository to your local machine.

Open in Visual Studio Code: Open the cloned repository in Visual Studio Code.

Run the Script: Open a terminal in Visual Studio Code and run the script using the following command:



Copy code
python your_script.py
Replace your_script.py with the filename of the Python script containing the face detection code.



Input Folder Path: Enter the path to the folder containing images when prompted.

View Output: The cropped images will be saved in an "output" folder in the same directory as the input images.




Requirements
Python 3.x
OpenCV
Pre-trained face detection model
Feel free to customize this README file according to your project's specifics and add any additional instructions or information as needed.







