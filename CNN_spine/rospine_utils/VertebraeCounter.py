import rospine_utils as utils
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image


class DlMethod(object):
    loader = None
    model = None
    transformation = None
    def __init__(self, model_path, loader_type = None, sweep_folder=None, sweep_label_folder=None, buffer_len=10,
                 sort_type=0, bias=0):

        self.loader_type = loader_type
        self.model_path = model_path
        self.buffer_len = buffer_len
        self.sweep_folder = sweep_folder
        self.sweep_label_folder = sweep_label_folder
        self.sort_type = sort_type
        self.bias = bias

        if loader_type is None:
            return

        if loader_type == "online":
            self.loader = utils.OnlineLoader()

        if loader_type == "offline":
            self.loader = utils.OfflineLoader(sweep_path=self.sweep_folder,
                                              sweep_label_path=self.sweep_label_folder,
                                              sort_type=self.sort_type,
                                              bias=self.bias)

    def load_model(self):
        self.model = utils.ModelLoader(self.model_path)
        self.model.to_device("cuda")
        self.model.eval()

    def release_model(self):
        self.model = None

    def process(self, input_data = None):
        return NotImplemented

    def set_loader(self, loader):
        self.loader = loader

    def set_input_transformation(self, transformation):
        self.transformation = transformation


class VertebraeCounter(DlMethod):

    def process(self, input_data = None):

        print(input_data.shape)
        current_image, current_label = Image.fromarray(input_data.astype('uint8')), -1
        if current_image is None:
            return None
        current_image = self.transformation(current_image).unsqueeze_(0)

        c1, c2 = self.processing_thread(self.model, [current_image], [current_label])
        return c1

    @staticmethod
    def plot(i1, i2, image):
        plt.clf()
        plt.subplot(1, 3, 1)
        plt.plot(i1)
        plt.subplot(1, 3, 2)
        plt.plot(i2)
        plt.subplot(1, 3, 3)
        plt.imshow(np.squeeze(image[0, :, :]), cmap='gray')
        plt.pause(0.001)

    @staticmethod
    def processing_thread(model, input_list, label_list):
        inputs, labels = utils.list2batch(input_list, label_list)
        output = model.run_inference(inputs.to("cuda"))
        prob = torch.sigmoid(output)

        c1 = prob[0, 0]
        c2 = prob[0, 1]

        c1_array = np.squeeze(c1.to("cpu").numpy())
        c2_array = np.squeeze(c2.to("cpu").numpy())

        return c1_array.tolist(), c2_array.tolist()
