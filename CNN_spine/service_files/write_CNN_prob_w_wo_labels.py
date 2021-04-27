### FIND the probabilities of Classes, "Gap", "NonGap"
### p_c1 - non Gap
### p_c2 - Gap

import rospine_utils as utils
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas
import os
from torch import nn

patients = []


def processing_thread(model, input_list, label_list):
    inputs, labels = utils.list2batch(input_list, label_list)
    output = model.run_inference(inputs.to("cpu"))

    # with torch.no_grad():
    #     output = model.forward(inputs)

    prob = torch.sigmoid(output)

    c1 = prob[0, 0]
    c2 = prob[0, 1]

    c1_array = np.squeeze(c1.to("cpu").numpy())
    c2_array = np.squeeze(c2.to("cpu").numpy())

    if labels is not None:
        labels = float(labels.to("cpu").item())
    else:
        return c1_array.tolist(), c2_array.tolist()


    return c1_array.tolist(), c2_array.tolist(), labels

def open_file(name_file):
    train_file = open(name_file, "r")
    lines_train = train_file.read().splitlines()
    print('patient names:',lines_train)
    return lines_train

def LoadModel(model_path):

    model = utils.ModelLoader_types(model_path, model_type="classification")
    model.to_device("cpu")

    model.eval()
    return  model

##############___Load model ##############

model_path = "/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/models/spinous_best_18_retrain.pt"
path = "/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/PolyU_sweep"
Labels_exist = True

model = LoadModel(model_path)

transformation = transforms.Compose([transforms.Resize(256),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
model.eval()

###########____Load_Patients ######


# patients = open_file('/home/maria/IFL_project/forceultrasuondspine/Spinous_counting/dataset_to_plot.txt')


# for patient in os.listdir(os.path.join(path,"images")):
#     patients.append(patient)



patients = 1
for i in range(patients):

    # patient_name = patients[i]
    loader_type = "offline"

    buffer_len = 1
    data_path = path
    print(data_path)
    sweep_folder = os.path.join(path,'Images')
    print(sweep_folder)
    if Labels_exist:
        sweep_label_folder = os.path.join(path,'Labels')
    else:
        sweep_label_folder = None
    sort_type = 0
    bias = 0
    # save_folder = os.path.join(path,'csv')
    pd_frame = pandas.DataFrame(columns=['CNN Probability', 'Label'])

    loader = utils.OfflineLoader(sweep_path=sweep_folder,
                                sweep_label_path=sweep_label_folder,
                                sort_type=sort_type,
                                bias=bias)


    # Initializing processing thread
    tensor_list, label_list = [], []
    p_c1, p_c2 = [], []
    plt.ion()


    while True:

        current_image, current_label = loader.get_next_image()
        if current_image is None:
            break
        tensor_list.append(transformation(current_image).unsqueeze_(0))
        if current_label is None:
            label_list = None
        label_list.append(current_label)

        if len(tensor_list) >= buffer_len:
            c1, c2, labels_list = processing_thread(model, tensor_list, label_list)
            p_c1.append(c1)
            p_c2.append(c2)


            pd_frame = pd_frame.append({'CNN Probability': c1,'Label': labels_list},
                                       ignore_index=True)


            pd_frame.to_csv(os.path.join(data_path, "CNN_Probability_label.csv"))


            # print(p_c1)
            # print(label_list)
            tensor_list, label_list = [], []

    # print(patient_name,'  ready')