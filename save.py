import numpy as np
from matplotlib import pyplot as plt
from torchvision import utils
import torch

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

color_map = np.array(
    [[0, 0, 0], [189, 16, 224], [80, 227, 194],
     [208, 2, 27], [248, 231, 28], [126, 211, 33],
     [134, 147, 209], [74, 144, 226]]
)

def save_img(outputs_list):

    # SAVE OUTPUT IMAGES
    for i in range(len(outputs_list)):
        outputs = outputs_list[i]
        #print("outputs before color_map:" + str(outputs))
        outputs_c = [color_map[i] for i in outputs.cpu()]
        for j in range(len(outputs_c)):
            output = outputs_c[j]/255
            #print(output)
            output = torch.from_numpy(output)
            output = output.permute(2,0,1)
            #print(output)
            #print(output.dtype)
            utils.save_image(output, './outputs/output'+str(i)+str(j)+'.png')