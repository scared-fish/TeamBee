Getting a UNet Model decide on the contents of beecells with images of an active beehive.

Images have to be stored in './imgs/inputs'  
Masks for training have to be stored in './imgs/masks'

Validated output by the UNet Model will be stored in './outputs' as .png files, aswell as a plot of statistics tracked during runtime.

Files and purposes:  
data.py:		Dataset initialization and using functions from transforms.py to edit the incoming images.  
evaluation.py:	Calculation of Dice-Coefficient and Accuracy.  
main.py:		Program start and setting of HyperParameters and devices used by the model.  
model.py:		UNet Model functions and workflow.  
save.py:		Saving the validation Images to our output folder.  
train.py:		Using the model to train on the data.  
transforms.py:	Contains functions to edit input images for more variety.  
validate.py:	Using evaluation.py and getting the data to use the functions.
