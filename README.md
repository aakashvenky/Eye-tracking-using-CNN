# Eye-tracking-using-CNN
 Eye Gesture Classification. Classified into three categories: whether seeing right, left or a eye-blink, with expectation of around 90% average accuracy by best models.
Introduction
Our goal will be to implement the Machine + Deep Learning(CNN) algorithms for Eye Gesture Classification. Classified into three categories: whether seeing right, left or a eye-blink, with expectation of around 90% average accuracy by best models.

Literature Review:
Eye gesture recognition using ML algorithms such as KNN and SVM to classify eye gestures into four categories: right, left, up, and down. But the dataset consisted of only 140 samples, and the best accuracy achieved was 90% using the KNN, and there was no blink classification, and outliers was not predicted by the model accurately.

image

Dataset used: EYE (Kaggle)
The Eye-Dataset is a collection of 14,997 JPEG images of eyes with annotations in XML format. The annotations provide information about the position and size of the iris, pupil, and eyelids. The images have different sizes and resolutions and were collected from various sources, including the internet and medical databases.

Problem definition
There exists an unparalleled connection between the body gestures and communication and researchers have tried to utilize the information contained in gestures to better optimize our understanding of messages being conveyed. Significant research has been done to decode the information contained in facial cues and hand gestures but one area that remains to be explored further is eye movement recognition. Detecting eye movement could have potentially life-changing applications and so we aim to use Machine Learning to detect eye movement patterns.

Dataset Description
Our Dataset consists of eye images of various gestures (left,right,center) acquired from Kaggle dataset and also created by us using haar cascade code (which detects the eye coordinate with the corresponding width (w) and height (h) of a rectangle, cropped from the moving live video data). We combine the kaggle dataset as well as our private dataset to create data for our model training and testing.

The images are normalized into 0-1 pixels for better computational, and histogram equivalization is used to prevent unwanted lighting/noise. It makes it uniform lighting (clear). This is one of our data-cleaning steps.
