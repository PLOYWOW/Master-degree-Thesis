import pandas as pd

n = ['image', 'blank1', 'state', 'blank2', 'truth_x', 'truth_y', 'blank3', 'estimate_x', 'estimate_y', 'blank4', 'error']
df = pd.read_csv('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Result Files/AlexNet/result.csv',header = None,names=n)

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

dfw = pd.DataFrame(image_cut, columns = ['file_path'])
dfw.to_csv("C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/File Path/file_path.csv")