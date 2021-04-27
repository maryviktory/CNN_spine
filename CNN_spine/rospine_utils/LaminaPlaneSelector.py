from rospine_utils import utils, DlMethod
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image


class LaminaPlaneSelector(DlMethod):

    def process(self, input_data = None):
        image_batches = self.get_batches_list(input_data, self.transformation, self.buffer_len)

        prob_vector = []
        for i, image_batch in enumerate(image_batches):
            c1 = self.processing_thread(self.model, image_batch)
            prob_vector.extend(c1)

        plt.plot(prob_vector)
        plt.show()

        #TODO: returns the image corresponding to the plane
        return np.zeros([100, 100, 3])

    @staticmethod
    def processing_thread(model, inputs):

        output = model.run_inference(inputs.to("cuda"))
        prob = torch.sigmoid(output)
        c1_array = np.squeeze(prob.to("cpu").numpy())

        return c1_array.tolist()

    @staticmethod
    def get_batches_list(image_list, transformation, batch_size):

        batches_list, label_batch_list = [], []
        tensor_list, tensor_labels_list = [], []
        for i, item in enumerate(image_list):
            pil_image = Image.fromarray(item.astype('uint8'))
            tensor_list.append(transformation(pil_image).unsqueeze_(0))

            if (i + 1) % batch_size == 0 or i == len(image_list) - 1:
                current_batch, current_label = utils.list2batch(tensor_list, tensor_labels_list)
                batches_list.append(current_batch)
                tensor_list = []
        return batches_list

    def plot(self, tensor_list, prob_list, idx):
        plt.ion()
        in_array = tensor_list.numpy()
        splitted_array = np.split(in_array, in_array.shape[0], axis= 0)

        batch_size = len(splitted_array)

        for i, image in enumerate(splitted_array):
            image_1c = np.squeeze(image[0, 2, :, :])
            print(image_1c.shape)
            prob = prob_list[0:i]

            print(idx, " ", idx + i + 1)
            a = np.arange(idx*batch_size, idx*batch_size + i, 1)
            print(len(a))
            if i == 0:
                continue

            plt.subplot(1, 2, 1)
            plt.plot(a, prob)

            plt.subplot(1, 2, 2)
            plt.imshow(image_1c, cmap='gray')
            plt.pause(0.1)