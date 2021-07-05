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

def save_img(outputs_list):

    for batch_index, (outputs, labels, original_images) in enumerate(outputs_list):
        if batch_index > 1:
            break
        for sample_index, (output, label, original_image) in enumerate(zip(outputs, labels, original_images)):
            if sample_index > 100:
                break
            output = color_map(output)
            label = color_map(label)
            label = np.squeeze(label)
            #original_image = (original_image + 1.0) / 2.0
            original_image = color_map_gray(original_image[0])

            output = np.concatenate((output, original_image, label), axis=1)
            plt.imsave('./outputs/output' + str(batch_index) + "_" + str(sample_index) + '.png', output)

    # SAVE OUTPUT IMAGES
    #for i in range(len(outputs_list)):
    #    outputs = outputs_list[i]
    #    #print("outputs before color_map:"  str(outputs))
    #    outputs_c = [color_map[i] for i in outputs.cpu()]
    #    for j in range(len(outputs_c)):
    #        output = outputs_c[j]/255
    #        #print(output)
    #        output = torch.from_numpy(output)
    #        output = output.permute(2,0,1)
    #        #print(output)
    #        #print(output.dtype)
    #        utils.save_image(output, './outputs/output'str(i)str(j)'.png')