import numpy as np
from matplotlib import pyplot as plt

color_map = plt.get_cmap("Set2", lut=9)
color_map_gray = plt.get_cmap("gray")

def save_img(outputs_list):

    for batch_index, (outputs, labels, original_images) in enumerate(outputs_list):
        if batch_index > 1:
            break
        for sample_index, (output, label, original_image) in enumerate(zip(outputs, labels, original_images)):
            if sample_index > 500:
                break
            output = color_map(output)
            label = color_map(label)
            label = np.squeeze(label)
            #original_image = (original_image + 1.0) / 2.0
            original_image = color_map_gray(original_image[0])

            output = np.concatenate((output, original_image, label), axis=1)
            plt.imsave('./outputs/output' + str(batch_index) + "_" + str(sample_index) + '.png', output)

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
    for i in range(15):
        pred_list.append(np.concatenate(tuple(predictions_whole[(i*20):(i*20)+20]),axis=1))
        label_list.append(np.concatenate(tuple(labels_whole[(i*20):(i*20)+20]),axis=1))
        img_list.append(np.concatenate(tuple(images_whole[(i*20):(i*20)+20]),axis=1))
    predictions_whole = np.concatenate(tuple(pred_list),axis=0)
    if epoch == 0:
        labels_whole = np.concatenate(tuple(label_list),axis=0)
        images_whole = np.concatenate(tuple(img_list),axis=0)
        plt.imsave('./outputs/labels_whole.png', labels_whole)
        plt.imsave('./outputs/images_whole.png', images_whole)
    else:
        plt.imsave('./outputs/predictions_whole' + str(epoch) + '.png', predictions_whole)