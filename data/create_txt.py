import os

out_file = open('brats2021/valid.txt', 'w', encoding='utf-8')

for i in os.listdir("/mnt/data/data/BraTs2021/RSNA_ASNR_MICCAI_BraTS2021_ValidationData/"):
    out_file.writelines("/mnt/data/data/BraTs2021/RSNA_ASNR_MICCAI_BraTS2021_ValidationData/" + i)
    out_file.writelines("\n")
out_file.close()
