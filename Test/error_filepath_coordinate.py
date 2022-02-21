import pandas as pd

name = ['image', 'blank1', 'state', 'blank2', 'truth_x', 'truth_y', 'blank3', 'estimate_x', 'estimate_y', 'blank4', 'error']
df = pd.read_csv('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Result Files/AlexNet/result.csv',header = None,names=name)

image = list(df["image"])
image_cut = []

for i in range(len(image)):
    count = 0
    keep = ''
    for j in image[i]:
        if count >= 14:
            keep = keep + j
        count += 1
    image_cut.append(keep)

model_name = ['VGG16+Dropout','VGG16','VGG16+Batch Normalization','Xception+Dropout','Xception','AlexNet']

VGG16_Dropout = {
    'file_path' : image_cut,
    'truth_x' : [],
    'truth_y' : [],
    'estimate_x' : [],
    'estimate_y' : []
}

VGG16 = {
    'file_path' : image_cut,
    'truth_x' : [],
    'truth_y' : [],
    'estimate_x' : [],
    'estimate_y' : []
}

VGG16_BatchNormalization = {
    'file_path' : image_cut,
    'truth_x' : [],
    'truth_y' : [],
    'estimate_x' : [],
    'estimate_y' : []
}

Xception_Dropout = {
    'file_path' : image_cut,
    'truth_x' : [],
    'truth_y' : [],
    'estimate_x' : [],
    'estimate_y' : []
}

Xception = {
    'file_path' : image_cut,
    'truth_x' : [],
    'truth_y' : [],
    'estimate_x' : [],
    'estimate_y' : []
}

AlexNet = {
    'file_path' : image_cut,
    'truth_x' : [],
    'truth_y' : [],
    'estimate_x' : [],
    'estimate_y' : []
}

model_dict = [VGG16_Dropout,VGG16,VGG16_BatchNormalization,Xception_Dropout,Xception,AlexNet]

for j in range(len(model_name)):
    df = pd.read_csv('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Result Files/'+model_name[j]+'/result.csv',header = None,names = name)
    truth_x = list(df["truth_x"])
    truth_y = list(df["truth_y"])
    estimate_x = list(df["estimate_x"])
    estimate_y = list(df["estimate_y"])
    model_dict[j]["truth_x"] = truth_x
    model_dict[j]["truth_y"] = truth_y
    model_dict[j]["estimate_x"] = estimate_x
    model_dict[j]["estimate_y"] = estimate_y

VGG16_Dropout = pd.DataFrame(VGG16_Dropout)
VGG16 = pd.DataFrame(VGG16)
VGG16_BatchNormalization = pd.DataFrame(VGG16_BatchNormalization)
Xception_Dropout = pd.DataFrame(Xception_Dropout)
Xception = pd.DataFrame(Xception)
AlexNet = pd.DataFrame(AlexNet)

dfw = pd.concat([VGG16_Dropout, VGG16, VGG16_BatchNormalization, Xception_Dropout, Xception, AlexNet],keys=['VGG16_Dropout', 'VGG16', 'VGG16_BatchNormalization', 'Xception_Dropout', 'Xception', 'AlexNet'])
dfw.to_csv("C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Code/Test/extract.csv")

print(1)
print(2)
print(3)
print(4)
print(5)
print(6)