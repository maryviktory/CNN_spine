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
from PIL import Image
from torch import nn

num_classes = 3

def processing_thread(model, input_list, label_list, classes):
    inputs, labels = utils.list2batch(input_list, label_list)
    output = model.run_inference(inputs.to("cpu"))

    # with torch.no_grad():
    #     output = model.forward(inputs)

    prob = torch.sigmoid(output)

    c1 = prob[0, 0]
    c2 = prob[0, 1]
    c1_array = np.squeeze(c1.to("cpu").numpy())
    c2_array = np.squeeze(c2.to("cpu").numpy())

    labels = float(labels.to("cpu").item())

    if classes > 2:
        c3 = prob[0,2]
        c3_array = np.squeeze(c3.to("cpu").numpy())
        return c1_array.tolist(), c2_array.tolist(), c3_array.tolist(), labels

    return c1_array.tolist(), c2_array.tolist(), labels

def open_file(name_file):
    train_file = open(name_file, "r")
    lines_train = train_file.read().splitlines()
    print('patient names:',lines_train)
    return lines_train



def get_label_from_image(label_image):
    label_array = np.asarray(label_image)
    pixel_sum = np.sum(label_array)

    return int(pixel_sum > 0)
##############___Load model ##############
def LoadModel(model_path):

    model = utils.ModelLoader_types(num_classes, model_path, model_type="classification")
    model.to_device("cpu")

    model.eval()
    return  model

if num_classes > 2:
    model_path = "/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/PWH_sweeps/model_best_resnet_fixed_False_pretrained_True_PolyU_dataset_3_classes.pt"
else:
    model_path = "/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/PWH_sweeps/model_best_resnet_fixed_False_pretrained_True_PolyU_dataset_2_classes.pt"

# model = models.resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 2)
#
# checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
# print(checkpoint)
# model.load_state_dict(checkpoint['model_state_dict'])


model = LoadModel(model_path)


transformation = transforms.Compose([transforms.Resize(256),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])


###########____Load_Patients ######


# patients = open_file('/home/maria/IFL_project/forceultrasuondspine/Spinous_counting/dataset_to_plot.txt')
path = "/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/PWH_sweeps/Subjects dataset/"

if num_classes>2:

    patients = ["sweep012","sweep018","sweep014","sweep015","sweep017","sweep019","sweep3006"]
else:
    patients = ["sweep012","sweep018","sweep013","sweep014","sweep015","sweep017","sweep019","sweep020","sweep3005","sweep3006"]
for i in range(len(patients)):

    patient_name = patients[i]
    loader_type = "offline"

    buffer_len = 1
    data_path = path
    sweep_folder = os.path.join(path,patient_name,'Images')
    print(sweep_folder)

    sweep_label_folder = os.path.join(path,patient_name,'Labels')
    sort_type = 0
    bias = 0
    save_folder = os.path.join(path,'csv')


    loader = utils.OfflineLoader(sweep_path=sweep_folder,
                                sweep_label_path=sweep_label_folder,
                                sort_type=sort_type,
                                bias=bias)
    if num_classes > 2:
        sweep_folder = os.path.join(path, patient_name, 'Images')
        sweep_label_folder = os.path.join(path, patient_name, 'Labels_sacrum')

        sort_type = 0
        bias = 0

        pd_frame = pandas.DataFrame(columns=['Spinous Probability','Sacrum', 'Label'])

        loader_sacrum = utils.OfflineLoader(sweep_path=sweep_folder,
                                     sweep_label_path=sweep_label_folder,
                                     sort_type=sort_type,
                                     bias=bias)

    else:
        pd_frame = pandas.DataFrame(columns=['Spinous Probability', 'Label'])
    # Initializing processing thread
    tensor_list, label_list = [], []
    sacrum_label_list = []
    p_c1, p_c2, p_c3= [], [],[]
    plt.ion()


    while True:

        if num_classes>2:
            current_image, current_label = loader.get_next_image()
            curr_img, sacrum_current_label = loader_sacrum.get_next_image()
            if current_image is None:
                break

            if curr_img is None:
                break
            tensor_list.append(transformation(current_image).unsqueeze_(0))
            label_list.append(current_label)
            sacrum_label_list.append(sacrum_current_label)
        else:
            current_image, current_label = loader.get_next_image()
            if current_image is None:
                break
            tensor_list.append(transformation(current_image).unsqueeze_(0))
            label_list.append(current_label)

        if len(tensor_list) >= buffer_len:

            if num_classes>2:
                # print(len(sacrum_label_list), len(tensor_list), len(label_list))
                # print(sacrum_label_list, label_list)
                c1, c2,c3, labels_list = processing_thread(model, tensor_list, label_list,num_classes)

                p_c1.append(c1)
                p_c2.append(c2)
                p_c3.append(c3)
                pd_frame = pd_frame.append({'Spinous Probability': c2, 'Sacrum': c3,'Label': labels_list},
                                           ignore_index=True)

                pd_frame["Sacrum_label"] = sacrum_label_list

                pd_frame.to_csv(os.path.join(data_path, patient_name + "_Spinous_Sacrum_Probability_label.csv"))

                tensor_list, label_list = [], []
            else:
                c1, c2, labels_list = processing_thread(model, tensor_list, label_list, num_classes)
                p_c1.append(c1)
                p_c2.append(c2)

                pd_frame = pd_frame.append({'Spinous Probability': c2, 'Label': labels_list},
                                       ignore_index=True)

                pd_frame.to_csv(os.path.join(data_path, patient_name+"_Spinous_Probability_label.csv"))


            # print(p_c1)
            # print(label_list)
                tensor_list, label_list = [], []

    # if num_classes > 2:
    #     sweep_label_folder = os.path.join(path, patient_name, 'Labels_sacrum')
    #     file_list = [file for file in os.listdir(sweep_label_folder) if ".png" in file]
    #
    #
    #     for image in file_list:
    #
    #         current_label = get_label_from_image(Image.open(os.path.join(sweep_label_folder,image)))
    #         pd_frame = pd_frame.append({'Label_sacrum': current_label},ignore_index=True)
    #
    print(patient_name,'  ready')