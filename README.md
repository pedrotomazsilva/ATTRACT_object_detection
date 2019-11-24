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
1. model configuration file
2. data loader
3. plot training graphs 

In order to run the code I provide a link to a google colab kernel I developed and which you are free to edit and run through this link:

`https://colab.research.google.com/drive/1hidNStmD3Jm2T_DhIaMi17mZWF-39_o2`

The data and cuda functions are provided through this link to my drive:

cuda functions: `https://drive.google.com/open?id=1EZH3WRzIrfiSFfAyTjOH-gBCYeywvbL7`

data: `https://drive.google.com/open?id=1amkl3Th8FXpd36KSwPODVlWXXPY-T99n`

To set up the files in order to run the google colab kernel you will have to create a directory in your own google drive called "Projetos". Then inside it you will have to upload the "medicaldetectiontoolkit_COLAB" folder with the cuda functions copied to "Projetos/medicaldetectiontoolkit_COLAB/" and the data copied to "Projetos/medicaldetectiontoolkit_COLAB/experiments/attract_exp/"

Some examples depicting the use of developed functions can be found in "attract_exp/using_scripts_examples". For bounding box creation you can check the instructions and functions provided in "attract_exp/bbox_utilities".

For more instructions you can contact me by email: pedrotomaz20@gmail.com or by phone +351912057823. 
