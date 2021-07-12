Getting a UNet Model decide on the contents of beecells with images of an active beehive.

Images have to be stored in './imgs/inputs'\n
Masks for training have to be stored in './imgs/masks'

Validated output by the UNet Model will be stored in './outputs' as .png files, aswell as a plot of statistics tracked during runtime.

Files and purposes:\n
data.py:		Dataset initialization and using functions from transforms.py to edit the incoming images.\n
evaluation.py:	Calculation of Dice-Coefficient and Accuracy.\n
main.py:		Program start and setting of HyperParameters and devices used by the model.\n
model.py:		UNet Model functions and workflow.\n
save.py:		Saving the validation Images to our output folder.\n
train.py:		Using the model to train on the data.\n
transforms.py:	Contains functions to edit input images for more variety.\n
validate.py:	Using evaluation.py and getting the data to use the functions.\n