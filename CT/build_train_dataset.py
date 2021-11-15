import os
import shutil

path = './images'
file_path = 'train_COVIDx-CT.txt'

with open(file_path, 'r') as f:
	data = f.readlines()

# NCP_list = data[5000:6000]
# CP_list = data[30000:31000]
# Normal_list = data[50000:51000]

NCP_list = data[3000:3350]
CP_list = data[32000:32350]
Normal_list = data[51000:51350]
# print(len(NCP_list))
# image_list = os.listdir(path)

# print(len(image_list))

# print(NCP_list[0].split()[0])
# src = os.path.join(path, NCP_list[0].split()[0])
# dst = os.path.join('./train', NCP_list[0].split()[0])
# shutil.copy(src, dst)

for i in NCP_list:
	file_name = i.split()[0]
	src = os.path.join(path, file_name)
	dst = os.path.join('./train_add', file_name)
	shutil.copy(src, dst)

for i in CP_list:
	file_name = i.split()[0]
	src = os.path.join(path, file_name)
	dst = os.path.join('./train_add', file_name)
	shutil.copy(src, dst)

for i in Normal_list:
	file_name = i.split()[0]
	src = os.path.join(path, file_name)
	dst = os.path.join('./train_add', file_name)
	shutil.copy(src, dst)