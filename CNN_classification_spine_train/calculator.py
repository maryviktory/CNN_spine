import numpy as np
import pandas as pd

def mean_from_file(file_path):
    frame = pd.read_csv(file_path)
    first = np.array(frame.loc[:, "Dist First Facet"])
    second = np.array(frame.loc[:, "Dist Second Facet"])
    first_mean = np.mean(first)
    first_std = np.std(first)
    second_mean = np.mean(second)
    second_std = np.std(second)
    facet_both = np.concatenate([first,second])

    print('first: ',first_mean, first_std)
    print('second: ', second_mean, second_std)
    print("mean: ",np.mean(facet_both), np.std(facet_both))







path = "/media/maryviktory/My Passport/IPCAI 2020 TUM/Facet_Db/test/Facet_Distance_4patients_5vert_each_bad_ones_3facets.csv"

mean_from_file(path)

# b = [0.897,	0.927, 0.645,0.704,0.914]
# # a = 0
# #
# s = np.mean(b)
# c = np.std(b)
# print(s,c)