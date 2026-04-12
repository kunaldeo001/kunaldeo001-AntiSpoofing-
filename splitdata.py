import os 
import random
import shutil
from itertools import islice

outputFolderPath = "Dataset/SplitData"
inputFolderpath = "Dataset/all"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ['fake', 'real']


if os.path.exists(outputFolderPath):
    shutil.rmtree(outputFolderPath)
os.makedirs(outputFolderPath, exist_ok=True)


for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(outputFolderPath, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(outputFolderPath, split, "labels"), exist_ok=True)


listNames = os.listdir(inputFolderpath)
uniqueNames = list(set(name.split('.')[0] for name in listNames))


random.shuffle(uniqueNames)


lenData = len(uniqueNames)
print(f'Total Images: {lenData}')

lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = int(lenData * splitRatio['test'])

# Adjust lengths to ensure total matches
if lenData != lenTrain + lenTest + lenVal:
    remaining = lenData - (lenTrain + lenTest + lenVal)
    lenTrain += remaining

print(f'Total Images: {lenData} \nSplit: {lenTrain} {lenVal} {lenTest}')

# Split data
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
print(f'Total Images: {lenData} \nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}')
print(f'Number of splits: {len(Output)}')


sequence = ['train', 'val', 'test']
for i, out in enumerate(Output):
    for fileName in out:
        src_image = os.path.join(inputFolderpath, f'{fileName}.jpg')
        src_label = os.path.join(inputFolderpath, f'{fileName}.txt')
        dest_image = os.path.join(outputFolderPath, sequence[i], 'images', f'{fileName}.jpg')
        dest_label = os.path.join(outputFolderPath, sequence[i], 'labels', f'{fileName}.txt')

        if os.path.exists(src_image):
            shutil.copy(src_image, dest_image)
        else:
            print(f"Warning: {src_image} does not exist.")

        if os.path.exists(src_label):
            shutil.copy(src_label, dest_label)
        else:
            print(f"Warning: {src_label} does not exist.")

print("Split Process Completed...")

# Create data.yaml file
dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'

with open(os.path.join(outputFolderPath, "data.yaml"), 'w') as f:
    f.write(dataYaml)

print("Data.yaml file Created...")

