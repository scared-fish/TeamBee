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