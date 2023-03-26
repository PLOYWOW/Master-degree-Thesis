#python main.py(0) VGG16(1) savefolder(./savemodel/~)(2) fp(1or9or17)(3) color(color/gray)(4) mode(5) databasepattern(6) cross(7) max(8)
#mode 0 = learning by leave one person out
#mode 1 = test by leave one person out
#mode 2 = Fine Tuning by leave one person out
#mode 3 = learning by all subjects
#mode 4 = Fine Tuning by all subjects
import os
import sys
import numpy as np
import pandas as pd
import numpy as np
import math
import cv2
import shutil
import random
from numpy import linalg as LA
from keras.utils import Sequence
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model
from PIL import Image
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Flatten
from keras.applications.vgg16 import VGG16
from keras import models
from keras import layers
from keras import optimizers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import def_func as func
import build_model as bd

#main###########################################################
print("********************************************")
modelname = ""
saveFileName = ""
mode = 0
fp = 0
color_mode = "gray"
databasepattern = ""
crossVNo = 0
crossVNoMax = 0

if(len(sys.argv)==9):
	print("read input")
	modelname = str(sys.argv[1])
	fp = int(sys.argv[3])
	color_mode = str(sys.argv[4])
	mode = int(sys.argv[5])
	databasePattern = str(sys.argv[6])
	crossVNo = int(sys.argv[7])
	crossVNoMax = int(sys.argv[8])
	if mode == 0 or mode == 1: #Cross-validation
		saveFilePath = str(sys.argv[2]) + "_" + color_mode + "_Cross_" + str(crossVNo)
	else:
		saveFilePath = str(sys.argv[2]) + "_" + color_mode + "_ini"
else:
	print("invalid parameter")
	exit()

trainlist = []
testlist = []
if mode == 3 or mode == 4:
	trainlist = [databasePattern % i for i in range(0,crossVNoMax)]
	#testlist = trainlist[crossVNo:crossVNo+1]
	#trainlist.remove(testlist[0])
	#testlist[0]=testlist[0].replace("_da","")
	print("Select file (Train) : %s" % (trainlist) )
	print("Select file (Test) : %s" % (testlist) )
else:
	trainlist = [databasePattern % i for i in range(0,crossVNoMax)]
	testlist = trainlist[crossVNo:crossVNo+1]
	trainlist.remove(testlist[0])
	testlist[0]=testlist[0].replace("_da","")
	testlist[0]=testlist[0].replace("roida","roi")
	print("Select file (Train) : %s" % (trainlist) )
	print("Select file (Test) : %s" % (testlist) )

print "model name : " + str(modelname)
print "savefile path : "+ str(saveFilePath)
print("feature point : " + str(fp))
print "mode : "+ str(mode)
print "database pattern : "+ str(databasePattern)
print "Cross validation : "+  str(crossVNo) + "/" + str(crossVNoMax)

if mode == 0 or mode == 3:
	if os.path.exists(saveFilePath):
		shutil.rmtree(saveFilePath)
	os.mkdir(saveFilePath)
	os.chmod(saveFilePath, 0777)

train_img, train_state, train_point = func.csv2list(trainlist, fp, "train")
if mode != 3 and mode != 4:
	test_img, test_state, test_point = func.csv2list(testlist, fp, "test")

#print(len(test_img))
#print(train_point)

train_gen = func.MyDataGenerator()
if mode != 3 and mode != 4:
	test_gen = func.MyDataGenerator()

#print(test_img)

epoch = 30
batch_size = 50

###############################################################################################
if mode == 0 or mode == 3: #mode 0 training by leave-one-person-out and mode 3 training by all subjects
	print("build model")
	model = bd.def_build_model(modelname, fp)

elif mode == 1: #mode 1 test by leave-one-person-out
	print("load model")
	model = load_model(saveFilePath + '/result.h5') #saveFilePath = str(sys.argv[2]) + "_" + color_mode + "_Cross_" + str(crossVNo)

elif mode == 2 or mode == 4: #mode 2 fine-tuning by leave-one-person-out and mode 4 fine-tuning by all subjects
	#print("build model")
	#model = bd.def_build_model(modelname, fp)
	print("load_model for fine tuning")
	model = load_model('./savemodel/Sakamoto_pretrained_VGG16_bn/result.h5') #saveFilePath = str(sys.argv[2]) + "_" + color_mode + "_ini"

	#change saveFilePath
	#index = saveFilePath.find('_All') + 4
	#saveFilePath = saveFilePath[:index] + '_ft_Cross_' + saveFilePath[index:]
	if mode == 2: #mode 2
		saveFilePath = saveFilePath.replace("_ini","") + "_ft_Cross_" + str(crossVNo)
	else: #mode 4
		saveFilePath = saveFilePath.replace("_ini","") + "_ft_All"

	if os.path.exists(saveFilePath):
		shutil.rmtree(saveFilePath)
	os.mkdir(saveFilePath)
	os.chmod(saveFilePath, 0777)

###############################################################################################
if mode == 0 or mode == 2 or mode == 3 or mode == 4: #without mode 1:test by leave-one-person-out
	#training
	print("training")
	if mode == 0 or mode == 2: #mode 0 training by leave one person out, mode 2 fine tuning by leave one person out
		history = model.fit_generator(
			generator = train_gen.flow_from_csv(allimg=train_img, allpoint=train_point, batch_size=batch_size, data_type="train", color_mode=color_mode, length=len(train_img)),
			epochs=epoch,
			steps_per_epoch=int(np.ceil(len(train_img) / batch_size)),
			validation_data = test_gen.flow_from_csv(allimg=test_img, allpoint=test_point, batch_size=batch_size, data_type="test", color_mode=color_mode, length=len(test_img)),
			validation_steps = int(np.ceil(len(test_img) / batch_size)),
			verbose = 1)

		#learning curve
		train_loss = history.history['loss']
		test_loss = history.history['val_loss']

		epochs = range(len(train_loss))

		plt.figure()

		plt.plot(epochs, train_loss, '-', label='Training loss')
		plt.plot(epochs, test_loss, '-', label='Test loss')
		plt.yscale('log')
		plt.title('Learning Curve')
		plt.xlabel("epoch")
		plt.ylabel("loss")
		plt.grid()
		plt.legend()

		plt.savefig(saveFilePath + "/learning_curve.png")

	elif mode == 3 or mode == 4: #mode 3 training by all subjects, mode 4 fine tuning by all subjects
		history = model.fit_generator(
			generator = train_gen.flow_from_csv(allimg=train_img, allpoint=train_point, batch_size=batch_size, data_type="train", color_mode=color_mode, length=len(train_img)),
			epochs=epoch,
			steps_per_epoch=int(np.ceil(len(train_img) / batch_size)),
			verbose = 1)
		#learning curve
		train_loss = history.history['loss']

		epochs = range(len(train_loss))

		plt.figure()

		plt.plot(epochs, train_loss, '-', label='Training loss')
		plt.yscale('log')
		plt.title('Learning Curve')
		plt.xlabel("epoch")
		plt.ylabel("loss")
		plt.grid()
		plt.legend()

		plt.savefig(saveFilePath + "/learning_curve.png")

	print("save model")
	model.save(saveFilePath + '/result.h5')

	#score = model.evaluate_generator(
	#	generator = test_gen.flow_from_csv(allimg=test_img, allpoint=test_point, batch_size=batch_size,data_type="test",  length = len(test_img)), steps=int(np.ceil(len(test_img) / batch_size)), max_queue_size=10,workers=1,use_multiprocessing=False)
	#print("test loss:"+ str(score[0]))
	#print("test acc:" + str(score[1]))

###############################################################################################
"""
result = model.predict_generator(
	generator = test_gen.flow_from_csv(allimg=test_img, allpoint=test_point, batch_size=1, data_type="test", color_mode=color_mode, length=len(test_img)), steps=int(np.ceil(len(test_img) / batch_size)), max_queue_size = 10, workers = 1, use_multiprocessing=False, verbose=1)
"""
if mode != 3 and mode != 4: #for mode 0:train by lopo, 1:test by lopo, 2:train ft lopo #mode 3 training by all subjects, mode 4 fine tuning by all subjects 
	result = model.predict_generator(
		generator = test_gen.flow_from_csv(allimg=test_img, allpoint=test_point, batch_size=1, data_type="test", color_mode=color_mode, length=len(test_img)), steps=len(test_img), max_queue_size = 10, workers = 1, use_multiprocessing=False, verbose=1)
	print(result)

	################################################################################################

	#output to result.csv
	print("output to result.csv")
	df_r = pd.DataFrame(test_img)
	df_r['space_img_esgt']=''
	df_esgt = pd.DataFrame(test_state)
	df_r = pd.concat([df_r,df_esgt],axis=1)
	df_r['space_esgt_ptsgt'] = ''
	point_gt_arr = np.array(test_point)
	df_ptsgt = pd.DataFrame(point_gt_arr)
	df_ptsgt = df_ptsgt.astype(float)
	df_r = pd.concat([df_r,df_ptsgt],axis=1)
	df_r['space_ptsgt_ptsest'] = ''
	df_ptsest = pd.DataFrame(result)
	df_ptsest = df_ptsest.astype(float)
	df_r = pd.concat([df_r,df_ptsest],axis=1)

	x = 0
	y = 0
	l_dis = []
	l_temp = []

	for i in range(0,len(df_ptsest)):
		for j in range(0,len(df_ptsest.columns),2):
			#print(i,j)
			x = abs(df_ptsgt.iat[i,j] - df_ptsest.iat[i,j])
			y = abs(df_ptsgt.iat[i,j+1] - df_ptsest.iat[i,j+1])
			d = np.sqrt(x*x+y*y)
			l_temp.append(d)
		l_dis.append(l_temp)	
		l_temp = []

	df_dis = pd.DataFrame(l_dis)
	df_r['space_ptsest_dis'] = ''
	df_r = pd.concat([df_r, df_dis],axis=1)

	df_r.to_csv(saveFilePath + "/result.csv",index=False,header=False)

	################################################################################################

	#output to classified.csv
	if fp == 8 or fp == 9 or fp == 16 or fp == 17:
		print("output to classified.csv")
		df_e = pd.DataFrame(test_img)
		df_e['space_img_esgt']=''
		df_e = pd.concat([df_e,df_esgt],axis=1)

		anglelist, esestlist = func.open_close_judge(df_ptsest, fp)
	
		df_e['space_esgt_angle']=''
		df_angle = pd.DataFrame(anglelist)
		df_e = pd.concat([df_e,df_angle],axis=1)
		df_e['space_angle_esest']=''
		df_esest = pd.DataFrame(esestlist)
		df_e = pd.concat([df_e,df_esest],axis=1)

		totallist = []
		openlist = []
		closelist = []
		#cm_gt_est_list, 0:open, 1:close
		cm_00_list = []
		cm_01_list = []
		cm_10_list = []
		cm_11_list = []
		df_bool = (df_esgt == 0)
		open_num = df_bool.sum()
		open_num = open_num.values[0]
		df_bool = (df_esgt == 1)
		close_num = df_bool.sum()
		close_num = close_num.values[0]
		print("open_num : " + str(open_num))
		print("close_num : " + str(close_num))

		for i in range(0,91):
			cm_00_sum = 0
			cm_01_sum = 0
			cm_10_sum = 0
			cm_11_sum = 0

			for j in range(0,len(df_esgt)):
				if df_esgt.iat[j,0] == df_esest.iat[j,i]:
					if df_esgt.iat[j,0] == 0:
						cm_00_sum = cm_00_sum + 1
					elif df_esgt.iat[j,0] == 1:
						cm_11_sum = cm_11_sum + 1
				else:
					if df_esgt.iat[j,0] == 0:
						cm_01_sum = cm_01_sum + 1
					elif df_esgt.iat[j,0] == 1:
						cm_10_sum = cm_10_sum + 1
				

			totallist.append(float(cm_00_sum+cm_11_sum)/len(df_esgt))
			openlist.append(float(cm_00_sum)/open_num)
			closelist.append(float(cm_11_sum)/close_num)
			cm_00_list.append(cm_00_sum)
			cm_01_list.append(cm_01_sum)
			cm_10_list.append(cm_10_sum)
			cm_11_list.append(cm_11_sum)

		df_e['space_esest_acc']=''
		thlist = list(range(0,91))
		df_th = pd.DataFrame(thlist)
		df_e = pd.concat([df_e,df_th],axis=1)
		df_total = pd.DataFrame(totallist)
		df_e = pd.concat([df_e,df_total],axis=1)
		df_open = pd.DataFrame(openlist)
		df_e = pd.concat([df_e,df_open],axis=1)
		df_close = pd.DataFrame(closelist)
		df_e = pd.concat([df_e,df_close],axis=1)

		df_bool = (df_total[0] == df_total[0].max())
		total_index_list = list(df_total[0][df_bool].index) #total_index_list = appropriate angle
		df_e.insert(3,'best acc',df_esest.iloc[:, total_index_list[len(total_index_list)/2]])

		df_e.to_csv(saveFilePath + "/classified.csv",index=False,header=False)

		#draw graph using thlist,totallist,openlist,closelist
		graph = plt.figure()
		plt.xlabel("degree")
		plt.ylabel("acc")
		plt.plot(thlist,totallist,color="blue",label="total")
		plt.plot(thlist,openlist,color="red",label="open")
		plt.plot(thlist,closelist,color="green",label="close")
		plt.rcParams['figure.subplot.right'] = 0.5
		plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0, fontsize=8)
		plt.grid()
		plt.savefig(saveFilePath + "/graph.png",bbox_inches='tight')
		plt.close(graph)

	################################################################################################
	#output to txt
	print("output to txt")
	inner_x = 0
	inner_y = 0
	inner_euc = 0
	outer_x = 0
	outer_y = 0
	outer_euc = 0
	allpts_x = 0
	allpts_y = 0
	allpts_euc = 0
	df_sub = pd.DataFrame()
	df_euc = pd.DataFrame()
	summary = ""
	aggregate = ",allpts_x,allpts_y,allpts_euc,allpts_euc_std,inner_x,inner_y,inner_euc,outer_x,outer_y,outer_euc,open_acc,close_acc,classification_acc,gt:open/est:open,gt:open/est:close,gt:close/est:open,gt:close/est:close,pcd_gt,pcd_gt_std,pcd_sys,pcd_sys_std,pcd_io,pcd_io_std,in_out_num\n"

	if fp == 8 or fp == 16:

		for i in range(0,fp*2):
			df_sub[i] = df_ptsgt[i] - df_ptsest[i]

		df_sub = np.abs(df_sub)

		sub_ave_list = []
		euc_ave_list = []

		for i in range(0,fp*2):
			sub_ave_list.append(df_sub[i].mean())
			if i < fp:
				euc_ave_list.append(df_dis[i].mean())
		
		for i in range(0, fp*2, 2):
			allpts_x = allpts_x + sub_ave_list[i]
			allpts_y = allpts_y + sub_ave_list[i+1]
			if ((i==0 or i==2 or i==(fp*2)-2) and fp==8) or ((i==0 or i==4 or i==(fp*2)-4) and fp==16):
				inner_x = inner_x + sub_ave_list[i]
				inner_y = inner_y + sub_ave_list[i+1]
			elif ((i==(fp-2) or i==fp or i==(fp+2)) and fp==8) or ((i==(fp-4) or i==fp or i==(fp+4)) and fp==16):
				outer_x = outer_x + sub_ave_list[i]
				outer_y = outer_y + sub_ave_list[i+1]

		for i in range(0, fp):
			allpts_euc = allpts_euc + euc_ave_list[i]
			if ((i==0 or i==1 or i==fp-1) and fp==8) or ((i==0 or i==2 or i==fp-2) and fp==16):
				inner_euc = inner_euc + euc_ave_list[i]
			elif ((i==fp/2-1 or i==fp/2 or i==fp/2+1) and fp==8) or ((i==fp/2-2 or i==fp/2 or i==fp/2+2) and fp==16):
				outer_euc = outer_euc + euc_ave_list[i]

		df_bool = (df_total[0] == df_total[0].max())
		total_index_list = list(df_total[0][df_bool].index) #total_index_list = appropriate angle

		efd_std = df_dis.stack().std()

		summary = summary + "allpts_x(efd) : " + str(allpts_x/fp) + "\n"
		summary = summary + "allpts_y(efd) : " + str(allpts_y/fp) + "\n"
		summary = summary + "allpts_euc(efd) : " + str(allpts_euc/fp) + "\n"
		summary = summary + "allpts_euc_std(efd) : " + str(efd_std) + "\n"
		summary = summary + "inner_x(efd) : " + str(inner_x/3) + "\n"
		summary = summary + "inner_y(efd) : " + str(inner_y/3) + "\n"
		summary = summary + "inner_euc(efd) : " + str(inner_euc/3) + "\n"
		summary = summary + "outer_x(efd) : " + str(outer_x/3) + "\n"
		summary = summary + "outer_y(efd) : " + str(outer_y/3) + "\n"
		summary = summary + "outer_euc(efd) : " + str(outer_euc/3) + "\n"
		summary = summary + "open acc(efd) : " + str(cm_00_list[total_index_list[len(total_index_list)/2]]/float(open_num)) + "\n"
		summary = summary + "close acc(efd) : " + str(cm_11_list[total_index_list[len(total_index_list)/2]]/float(close_num)) + "\n"
		summary = summary + "classification acc(efd) : " + str(df_total.max().values[0]) + " (angle is "
		for index in total_index_list:
			summary = summary + str(index) + " "
		summary = summary + "[deg])\n"
		summary = summary + "Confusion Matrix :\n"
		summary = summary + "      est\n"
		summary = summary + "     0    1\n"
		summary = summary + "g 0 {:^4} {:^4}\n".format(cm_00_list[total_index_list[len(total_index_list)/2]], cm_01_list[total_index_list[len(total_index_list)/2]])
		summary = summary + "t 1 {:^4} {:^4}\n".format(cm_10_list[total_index_list[len(total_index_list)/2]], cm_11_list[total_index_list[len(total_index_list)/2]])
		summary = summary + "pcd error (when ground truth is open) : NaN\n"
		summary = summary + "pcd error std (when ground truth is open) : NaN\n"
		summary = summary + "pcd error (when system judged open) : NaN\n"
		summary = summary + "pcd error std (when system judged open) : NaN\n"
		summary = summary + "pcd error (when pc go out of fp) : NaN\n"
		summary = summary + "pcd error std (when pc go out of fp) : NaN\n"
		summary = summary + "in_out_num : NaN\n"
		summary = summary + "in_out_list : NaN\n"

		aggregate = aggregate + saveFilePath + "," + str(allpts_x/fp) + "," + str(allpts_y/fp) + "," + str(allpts_euc/fp) + "," + str(efd_std) + "," + str(inner_x/3) + "," + str(inner_y/3) + "," + str(inner_euc/3) + "," + str(outer_x/3) + "," + str(outer_y/3) + "," + str(outer_euc/3) + "," +str(cm_00_list[total_index_list[len(total_index_list)/2]]/float(open_num)) + "," + str(cm_11_list[total_index_list[len(total_index_list)/2]]/float(close_num)) + "," + str(df_total.max().values[0]) + "," +str(cm_00_list[total_index_list[len(total_index_list)/2]]) + "," + str(cm_01_list[total_index_list[len(total_index_list)/2]]) + "," + str(cm_10_list[total_index_list[len(total_index_list)/2]]) + "," + str(cm_11_list[total_index_list[len(total_index_list)/2]]) + ",-,-,-,-,-,-,-"

		with open(saveFilePath + '/summary.txt','w') as f:
			f.write(summary)
		with open(saveFilePath + '/aggregate.csv','w') as f:
			f.write(aggregate)

	elif fp == 9 or fp == 17:
		for i in range(0,fp*2):
			df_sub[i] = df_ptsgt[i] - df_ptsest[i]

		df_sub = np.abs(df_sub)

		sub_ave_list = []
		euc_ave_list = []

		for i in range(0,fp*2):
			sub_ave_list.append(df_sub[i].mean())
			if i < fp:
				euc_ave_list.append(df_dis[i].mean())

		for i in range(0, fp*2-2, 2):
			allpts_x = allpts_x + sub_ave_list[i]
			allpts_y = allpts_y + sub_ave_list[i+1]
			if ((i==0 or i==2 or i==(fp-1)*2-2) and fp==9) or ((i==0 or i==4 or i==(fp-1)*2-4) and fp==17):
				inner_x = inner_x + sub_ave_list[i]
				inner_y = inner_y + sub_ave_list[i+1]
			elif ((i==(fp-1)-2 or i== fp-1 or i==(fp-1)+2) and fp==9) or ((i==(fp-1)-4 or i== fp-1 or i==(fp-1)+4) and fp==17):
				outer_x = outer_x + sub_ave_list[i]
				outer_y = outer_y + sub_ave_list[i+1]

		for i in range(0, fp-1):
			allpts_euc = allpts_euc + euc_ave_list[i]
			if ((i==0 or i==1 or i==fp-2) and fp==9) or ((i==0 or i==2 or i==fp-3) and fp==17):
				inner_euc = inner_euc + euc_ave_list[i]
			elif ((i==fp/2-1 or i==fp/2 or i==fp/2+1) and fp==9) or ((i==fp/2-2 or i==fp/2 or i==fp/2+2) and fp==17):
				outer_euc = outer_euc + euc_ave_list[i]

		df_bool = (df_total[0] == df_total[0].max())
		total_index_list = list(df_total[0][df_bool].index) 

		efd_std = df_dis.stack().std()

		#cal pcd
		#gt is open
		df_bool = (df_esgt[0] == 0)
		df_gt_pcd = df_dis.iloc[:,len(df_dis.columns)-1][df_bool]


		#sys judged open
		df_bool = (df_esest.iloc[:,total_index_list[len(total_index_list)/2]] == 0)
		df_sys_pcd = df_dis.iloc[:,len(df_dis.columns)-1][df_bool]
		in_out_list = func.in_out_judge(df_ptsest.iloc[:,0:len(df_ptsest.columns)-2], df_ptsest.iloc[:,len(df_ptsest.columns)-2:len(df_ptsest.columns)])
		df_in_out = pd.DataFrame(in_out_list)
		df_bool = (df_in_out[0] == 0)
		df_sys_in_out_pcd = df_sys_pcd[df_bool]
		print(df_sys_in_out_pcd)

		df_class = pd.read_csv(saveFilePath + "/classified.csv",header=None)
		df_class.insert(4,'io',df_in_out)
		df_class.to_csv(saveFilePath + "/classified.csv",index=False,header=False)
		df_in_out.to_csv(saveFilePath + "/in_out.csv",index=False,header=False)

		io_temp = ""
		io_sum = 0
		for i in range(0,len(in_out_list)):
			if (in_out_list[i] == 1) and (df_esest.iat[i,total_index_list[len(total_index_list)/2]] == 0):
				io_sum = io_sum + 1
				io_temp = io_temp + test_img[i] + "\n"


		summary = summary + "allpts_x(efd) : " + str(allpts_x/(fp-1)) + "\n"
		summary = summary + "allpts_y(efd) : " + str(allpts_y/(fp-1)) + "\n"
		summary = summary + "allpts_euc(efd) : " + str(allpts_euc/(fp-1)) + "\n"
		summary = summary + "allpts_euc_std(efd) : " + str(efd_std) + "\n"
		summary = summary + "inner_x(efd) : " + str(inner_x/3) + "\n"
		summary = summary + "inner_y(efd) : " + str(inner_y/3) + "\n"
		summary = summary + "inner_euc(efd) : " + str(inner_euc/3)+ "\n"
		summary = summary + "outer_x(efd) : " + str(outer_x/3) + "\n"
		summary = summary + "outer_y(efd) : " + str(outer_y/3) + "\n"
		summary = summary + "outer_euc(efd) : " + str(outer_euc/3) + "\n"
		summary = summary + "open acc(efd) : " + str(cm_00_list[total_index_list[len(total_index_list)/2]]/float(open_num)) + "\n"
		summary = summary + "close acc(efd) : " + str(cm_11_list[total_index_list[len(total_index_list)/2]]/float(close_num)) + "\n"
		summary = summary + "classification acc(efd) : " + str(df_total.max().values[0]) + " (angle is "
		for index in total_index_list:
			summary = summary + str(index) + " "
		summary = summary + "[deg])\n"
		summary = summary + "Confusion Matrix :\n"
		summary = summary + "      est\n"
		summary = summary + "     0    1\n"
		summary = summary + "g 0 {:^4} {:^4}\n".format(cm_00_list[total_index_list[len(total_index_list)/2]], cm_01_list[total_index_list[len(total_index_list)/2]])
		summary = summary + "t 1 {:^4} {:^4}\n".format(cm_10_list[total_index_list[len(total_index_list)/2]], cm_11_list[total_index_list[len(total_index_list)/2]])
		summary = summary + "pcd error (when ground truth is open) : " + str(df_gt_pcd.mean()) + "\n"
		summary = summary + "pcd error std (when ground truth is open) : " + str(df_gt_pcd.std()) + "\n"
		summary = summary + "pcd error (when system judged open) : " + str(df_sys_pcd.mean()) + "\n"
		summary = summary + "pcd error std (when system judged open) : " + str(df_sys_pcd.std()) + "\n"
		summary = summary + "pcd error (when pc go out of fp) : " + str(df_sys_in_out_pcd.mean()) + "\n"
		summary = summary + "pcd error std (when pc go out of fp) : " + str(df_sys_in_out_pcd.std()) + "\n"
		summary = summary + "in_out_num : " + str(io_sum) + "\n"
		summary = summary + "in_out_list : "
		if io_sum == 0:
			summary = summary + "NaN\n"
		else:
			summary = summary + "\n" + io_temp

		aggregate = aggregate + saveFilePath + "," + str(allpts_x/(fp-1)) + "," + str(allpts_y/(fp-1)) + "," + str(allpts_euc/(fp-1)) + "," + str(efd_std) + "," + str(inner_x/3) + "," + str(inner_y/3) + "," + str(inner_euc/3) + "," + str(outer_x/3) + "," + str(outer_y/3) + "," + str(outer_euc/3) + "," + str(cm_00_list[total_index_list[len(total_index_list)/2]]/float(open_num)) + "," + str(cm_11_list[total_index_list[len(total_index_list)/2]]/float(close_num)) + "," + str(df_total.max().values[0]) + "," + str(cm_00_list[total_index_list[len(total_index_list)/2]]) + "," + str(cm_01_list[total_index_list[len(total_index_list)/2]]) + "," + str(cm_10_list[total_index_list[len(total_index_list)/2]]) + "," + str(cm_11_list[total_index_list[len(total_index_list)/2]]) + "," + str(df_gt_pcd.mean()) + "," + str(df_gt_pcd.std()) + "," + str(df_sys_pcd.mean()) + "," + str(df_sys_pcd.std()) + "," + str(df_sys_in_out_pcd.mean()) + "," + str(df_sys_in_out_pcd.std()) + "," + str(io_sum)
	
		with open(saveFilePath + '/summary.txt','w') as f:
			f.write(summary)
		with open(saveFilePath + '/aggregate.csv','w') as f:
			f.write(aggregate)


	elif fp == 1:
		#cal pcd
		#gt is open
		df_bool = (df_esgt[0] == 0)
		df_gt_pcd = df_dis.iloc[:,0][df_bool]

		summary = summary + "allpts_x(efd) : NaN\n"
		summary = summary + "allpts_y(efd) : NaN\n"
		summary = summary + "allpts_euc(efd) : NaN\n"
		summary = summary + "allpts_euc_std(efd) : Nan\n"
		summary = summary + "inner_x(efd) : NaN\n"
		summary = summary + "inner_y(efd) : NaN\n"
		summary = summary + "inner_euc(efd) : NaN\n"
		summary = summary + "outer_x(efd) : NaN\n"
		summary = summary + "outer_y(efd) : NaN\n"
		summary = summary + "outer_euc(efd) : NaN\n"
		summary = summary + "open acc(efd) : NaN\n"
		summary = summary + "close acc(efd) : NaN\n"
		summary = summary + "classification acc(efd) : NaN\n"
		summary = summary + "Confusion Matrix :\n"
		summary = summary + "      est\n"
		summary = summary + "     0    1\n"
		summary = summary + "g 0 NaN  NaN\n"
		summary = summary + "t 1 NaN  NaN\n"
		summary = summary + "pcd error (when ground truth is open) : " + str(df_gt_pcd.mean()) + "\n"
		summary = summary + "pcd error std (when ground truth is open) : " + str(df_gt_pcd.std()) + "\n"
		summary = summary + "pcd error (when system judged open) : NaN\n"
		summary = summary + "pcd error std (when system judged open) : NaN\n"
		summary = summary + "pcd error (when pc go out of fp) : NaN\n"
		summary = summary + "pcd error std (when pc go out of fp) : NaN\n"

		summary = summary + "in_out_num : NaN\n"
		summary = summary + "in_out_list : NaN\n"

		aggregate = aggregate + saveFilePath + ",-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-," + str(df_gt_pcd.mean()) + "," + str(df_gt_pcd.std()) + ",-,-,-,-,-"
	
		with open(saveFilePath + '/summary.txt','w') as f:
			f.write(summary)
		with open(saveFilePath + '/aggregate.csv','w') as f:
			f.write(aggregate)







