import os

out_file_train = open('brats2020_new/train.txt', 'w', encoding='utf-8')
out_file_valid = open('brats2020_new/valid.txt', 'w', encoding='utf-8')
index = 0
for i in range(len(os.listdir("/media/dmia/data11/hh/MICCAI_BraTS2020_TrainingData/"))):
    if i < 300:
        out_file_train.writelines("/media/dmia/data11/hh/MICCAI_BraTS2020_TrainingData/" + os.listdir("/media/dmia/data11/hh/MICCAI_BraTS2020_TrainingData/")[i])
        out_file_train.writelines("\n")
    else:
        out_file_valid.writelines("/media/dmia/data11/hh/MICCAI_BraTS2020_TrainingData/" + os.listdir("/media/dmia/data11/hh/MICCAI_BraTS2020_TrainingData/")[i])
        out_file_valid.writelines("\n")
out_file_train.close()
out_file_valid.close()
