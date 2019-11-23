# ATTRACT_object_detection

This code is based on https://github.com/pfjaeger/medicaldetectiontoolkit 
It was adapted to fit the aim of detecting (or even segmenting) organels (nucleus and golgi) in endotelial cells' 3D images. 
The code developed for this specific task is located at experiments/attract_exp. Functions to read the raw czi images, preprocess them, anotate bounding boxes, divide the image into smaller partitions (cubes), visualize the images, plot training graphs are provided.
