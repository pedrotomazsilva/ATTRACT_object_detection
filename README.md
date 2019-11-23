# ATTRACT_object_detection

This code is based on https://github.com/pfjaeger/medicaldetectiontoolkit 
It was adapted to fit the aim of detecting (or even segmenting) organels (nucleus and golgi) in endotelial cells' 3D images.

The code developed for this specific task is located at experiments/attract_exp. 
Here I provide functions to:
1. read the raw czi images
2. preprocess them
3. divide the image into smaller partitions (cubes)
4. visualize the images
5. plot training graphs 
6. anotate bounding boxes
