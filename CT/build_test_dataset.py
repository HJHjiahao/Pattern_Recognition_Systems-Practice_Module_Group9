import os
import shutil

path = './images'
file_path = 'test_COVIDx-CT.txt'

with open(file_path, 'r') as f:
	data = f.readlines()

# NCP_list = data[1000:1500]
# CP_list = data[6000:6500]
# Normal_list = data[15000:15500]

NCP_list = data[2000:2200]
CP_list = data[7000:7200]
Normal_list = data[13000:13200]



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
	dst = os.path.join('./test_add', file_name)
	shutil.copy(src, dst)

for i in CP_list:
	file_name = i.split()[0]
	src = os.path.join(path, file_name)
	dst = os.path.join('./test_add', file_name)
	shutil.copy(src, dst)

for i in Normal_list:
	file_name = i.split()[0]
	src = os.path.join(path, file_name)
	dst = os.path.join('./test_add', file_name)
	shutil.copy(src, dst)