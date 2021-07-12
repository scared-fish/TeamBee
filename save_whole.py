import numpy as np
from matplotlib import pyplot as plt

# color_map = {
#      'Background': (0, 0, 0, 1)
#     'Bee': (189, 16, 224, 1),    #BD10E0
#     'Capped Honey Cell': (80, 227, 194, 1),  #50E3C2
#     'Empty Cell': (208, 2, 27, 1), #D0021B
#     'Larvae': (248, 231, 28, 1),  #F8E71C
#     'Pollen': (126, 211, 33, 1),  #7ED321
#     'Pupae': (134, 147, 209, 1), #8693D1
#     'Uncapped Honey Cell': (74, 144, 226, 1),  #4A90E2
# }

#color_map = np.array(
#    [[0, 0, 0], [189, 16, 224], [80, 227, 194],
#     [208, 2, 27], [248, 231, 28], [126, 211, 33],
#     [134, 147, 209], [74, 144, 226]]
#)

color_map = plt.get_cmap("Set2", lut=9)
color_map_gray = plt.get_cmap("gray")

def save_img_whole(outputs_list, epoch):
    predictions_whole = []
    labels_whole = []
    images_whole = []

    for batch_index, (outputs, labels, original_images) in enumerate(outputs_list):
        #if batch_index > 1:
        #    break
        for sample_index, (output, label, original_image) in enumerate(zip(outputs, labels, original_images)):
            #if sample_index > 300:
            #    break
            output = color_map(output)
            label = color_map(label)
            label = np.squeeze(label)
            original_image = color_map_gray(original_image[0])
            predictions_whole.append(output)
            labels_whole.append(label)
            images_whole.append(original_image)

            #output = np.concatenate((output, original_image, label), axis=1)
            #plt.imsave('./outputs/output' + str(batch_index) + "_" + str(sample_index) + '.png', output)
    
    pred_list= []
    label_list= []
    img_list= []
    for i in range(30):
        pred_list.append(np.concatenate(tuple(predictions_whole[(i*40):(i*40)+40]),axis=1))
        label_list.append(np.concatenate(tuple(labels_whole[(i*40):(i*40)+40]),axis=1))
        img_list.append(np.concatenate(tuple(images_whole[(i*40):(i*40)+40]),axis=1))
    predictions_whole = np.concatenate(tuple(pred_list),axis=0)
    if epoch == 0:
        labels_whole = np.concatenate(tuple(label_list),axis=0)
        images_whole = np.concatenate(tuple(img_list),axis=0)
        plt.imsave('./outputs/labels_whole.png', labels_whole)
        plt.imsave('./outputs/images_whole.png', images_whole)
    else:
        plt.imsave('./outputs/predictions_whole' + str(epoch) + '.png', predictions_whole)