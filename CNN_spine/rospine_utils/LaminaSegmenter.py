from rospine_utils import utils, DlMethod
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torchvision import  models
from torch import nn


class LaminaSegmenter(DlMethod):

    def process(self, input_data = None):

        if input_data is None:
            return input_data
        model = utils.ModelLoader(self.model_path)
        transformation = transforms.Compose([transforms.Resize(256),
                                             transforms.ToTensor()])

        #tensor_image = transformation(Image.fromarray(input_data, mode=None)).unsqueeze_(0)
        tensor_image = transformation(input_data).unsqueeze_(0)
        image_batch, _ = utils.list2batch([tensor_image], None)

        out = model.run_inference(image_batch)
        prob_tensor = torch.sigmoid(out['out'])
        p_map = np.squeeze(prob_tensor.to("cpu").numpy())

        segmentation = np.zeros(p_map.shape)
        segmentation[p_map > 0.5] = 255

        plt.subplot(1, 3, 1)
        plt.imshow(input_data)
        plt.subplot(1, 3, 2)
        plt.imshow(p_map*255)

        plt.show()

    @staticmethod
    def processing_thread(model, inputs, label_list):

        output = model.run_inference(inputs.to("cuda"))
        prob = torch.sigmoid(output['out']).float()

        p_map = np.squeeze(prob.to("cpu").numpy())

        segmentation = np.zeros(p_map.shape)
        segmentation[p_map > 0.5] = 255


        return p_map, segmentation
