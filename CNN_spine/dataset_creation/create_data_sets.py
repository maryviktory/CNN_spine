
import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d', '--dir', default="/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/PWH_sweeps/Subjects dataset", help='Database directory')
args = parser.parse_args()



def create_folder(folder,data_set,mode,clas):
    # folder = "/media/maria/My Passport/toNas/"
    data_folder= os.path.join(folder,'%s'%data_set)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    mode_folder= os.path.join(data_folder, "%s"%(mode))
    if not os.path.exists(mode_folder):
        os.mkdir(mode_folder)

    destination_folder = os.path.join(mode_folder, "%s" % (clas))
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    return destination_folder


def open_file(name_file):
    train_file = open(name_file, "r")
    lines_train = train_file.read().splitlines()
    print('patient names:',lines_train)
    return lines_train

# train_file = open_file("dataset_run5_training.txt")
# val_file = open_file("dataset_run5_validation.txt")
# test_file = open_file('dataset_run_test.txt')

train_list = ["sweep017",
"sweep5007",
"sweep3006",
'sweep012',
"sweep013",
"sweep014",
"sweep015",
"sweep019",
"sweep020",
"sweep3005",
"sweep5001",
"sweep3001"
]
val_list = ["sweep3000",
"sweep9004",
"sweep10001"
]
test_list = ["sweep018","sweep3013","sweep5005", "sweep9001"]
sacrum_exist = True

def copy_files (data_set, file,mode):

    for i in file:

        Gap_dir = os.path.join(os.path.join(args.dir, "%s"%i,'Classes','Gap'))
        data_train_gap = create_folder(args.dir,'%s'%data_set, '%s'%mode, 'Gap')
        print('folder of the patient',Gap_dir)
        NonGap_dir = os.path.join(os.path.join(args.dir, "%s"%i,'Classes','NonGap'))
        data_train_nongap = create_folder(args.dir,'%s'%data_set, '%s'%mode, 'NonGap')

        for filename in os.listdir(Gap_dir):
            full_file_name = os.path.join(Gap_dir, filename)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, data_train_gap)
            # print('filenames in Gap',filename)

        for filename in os.listdir(NonGap_dir):
            full_file_name = os.path.join(NonGap_dir, filename)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, data_train_nongap)
            # print(filename,'filenames in NonGap')

        if sacrum_exist:
            Sacrum_dir = os.path.join(os.path.join(args.dir, "%s" % i, 'Classes', 'Sacrum'))
            data_train_sacrum = create_folder(args.dir, '%s' % data_set, '%s' % mode, 'Sacrum')
            if os.path.exists(Sacrum_dir) and len(os.listdir(Sacrum_dir)) >0:
                for filename in os.listdir(Sacrum_dir):
                    full_file_name = os.path.join(Sacrum_dir, filename)
                    if os.path.isfile(full_file_name):
                        shutil.copy(full_file_name, data_train_sacrum)
                    # print(filename, 'filenames in Sacrum')
            else:
                continue

copy_files('data_19subj_2',train_list,'train')
# copy_files('data_19subj_2',val_list,'val')
# copy_files('data_19subj_2',test_list,'test')