from PIL import Image
import numpy as np
import os
import cv2
import logging


def get_loader(loader_name, sweep_path, sweep_label_path = None, video_idx = 0):
    if loader_name == "offline":
        return OfflineLoader(sweep_path, sweep_label_path)

    elif loader_name == "video":
        return VideoLoader(video_idx)
    elif loader_name == "online":
        return OnlineLoader()
    else:
        raise Exception("Can't load loader -- loader name not in the available list")


class StreamLoader(object):
    def get_next_image(self):
        return NotImplemented

    def load_full_sweep(self):
        return NotImplemented


class OfflineLoader(StreamLoader):
    images_list = []
    labels_list = []
    def __init__(self, sweep_path, sweep_label_path=None, sort_type = 1, bias = 0):

        self.bias = bias
        self.idx = bias
        self.is_labeled = sweep_label_path is not None
        self.get_image_list(sort_type, sweep_path, sweep_label_path)
        self.sweep_length = len(self.images_list)

        logging.info("Offline loader initialized: \nPath: {}\nsweep length: {}\nbias: {}"
                     .format(sweep_path, self.sweep_length, self.bias))

    def get_next_image(self):
        if self.idx >= self.sweep_length:
            return None, None

        current_image_path = self.images_list[self.idx]
        current_image = Image.open(current_image_path)
        if current_image.mode == 'L':
            current_image = current_image.convert(mode='RGB')

        if self.is_labeled:
            current_label_path = self.labels_list[self.idx]
            current_label = self._get_label_from_image(Image.open(current_label_path))
        else:
            current_label = -1

        self.idx += 1
        return current_image, current_label


    def load_full_sweep(self):
        image_list, label_list = [], []
        for i in range(self.bias, self.sweep_length):
            image, label = self.get_next_image()
            image_list.append(image)
            if self.is_labeled:
                label_list.append(label)

        return image_list, label_list

    def get_image_list(self, sort_type, sweep_path, sweep_label_path):
        if sort_type == 0:
            self.images_list = [os.path.join(sweep_path, item) for item in os.listdir(sweep_path)]
            if self.is_labeled:
                self.labels_list = [os.path.join(sweep_label_path, item) for item in os.listdir(sweep_label_path)]

            self.images_list.sort()
            self.labels_list.sort()

        elif sort_type == 1:
            self.images_list = [os.path.join(sweep_path, "image" + str(i) + ".png") for i in range(0, self.sweep_length)]
            if self.is_labeled:
                self.labels_list = [os.path.join(sweep_label_path, "image" + str(i) + ".png") for i in
                                    range(0, self.sweep_length)]

    def stream_sweep(self):
        print(self.images_list)
        for image in range(self.bias, self.sweep_length):
            pil_image, _ = self.get_next_image()

            cv2.imshow("", np.asarray(pil_image))
            cv2.waitKey(0)

    @staticmethod
    def _get_label_from_image(label_image):
        label_array = np.asarray(label_image)
        pixel_sum = np.sum(label_array)

        return int(pixel_sum > 0)


class VideoLoader(StreamLoader):
    cap = None

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    def get_next_image(self):
        ret, frame = self.cap.read()

        if not ret:
            return None, None

        return Image.fromarray(frame), None


class OnlineLoader(StreamLoader):
    def __init__(self):
        pass
