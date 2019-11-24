# ATTRACT_object_detection

This code is based on https://github.com/pfjaeger/medicaldetectiontoolkit 
It was adapted to fit the aim of detecting (or even segmenting) organels (nucleus and golgi) in endotelial cells' 3D images.

The code developed for this specific task is located at experiments/attract_exp. 

Several scripts/functions were developed:

For image/data handling and visualization:
1. read the raw czi images
2. preprocess them
3. divide the image into smaller partitions (cubes)
4. visualize the images
5. anotate bounding boxes

For the deep learning model:
1. configuration file
2. data loader
3. plot training graphs 

In order to run the code I provide a link to a google colab kernel I developed and which you are free to edit and run through this link:
`https://colab.research.google.com/drive/1hidNStmD3Jm2T_DhIaMi17mZWF-39_o2`
