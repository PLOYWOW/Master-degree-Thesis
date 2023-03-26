import sys
import os
import csv
import re

#python create_DB_pts.py img&asd&pts_foldername(ex.kan1) #Not use

databasefolder = "./VDO_resize/trial5/" #For writting image path in database csv file
# databasefolder = ""

subject_index = []
for i in range(1,7):
	for j in ("t","b","l","r"):
		subject_index.append("s00"+str(i)+"-"+j)
# print(subject_index)

for i in range(len(subject_index)):
# for i in range(1):
	# print(subject_index[i])
	folder_path = "./" + subject_index[i]  #For reading asd and pts file
	# folder_path = "/home/slab/ALS/Database/reannotate/result_from_assign/Matthew/example/0_l"
	output_path = "./efd_database_plusremaining/" + "efd_17pts_"+ subject_index[i] + ".csv" #For saving database csv file
	# output_path = "./eye_contour_database/" + "efd_17pts_matthew_example.csv"

	# subject_index = sorted(os.listdir(subject_folder))
	msg = ""
	# xratio = 120/600.0
	xratio = 1
	# yratio = 80/400.0
	yratio = 1

	# subject_index = sorted(os.listdir(subject_folder))

	for curdir, dirs, files in os.walk(folder_path):
		dirs.sort()
		# print(dirs)
		files.sort()
		# print(files)
		for imgfile in sorted(files):
			# print(imgfile)
			if imgfile.endswith(".asf"):
				asf = os.path.join(curdir,imgfile)
				jpg = asf.replace(".asf", ".jpg")
				pts = asf.replace(".asf", ".pts")
				#num_lines = sum(1 for line in open(pts_name, 'r')
				if os.path.exists(pts):
					with open(pts, 'r') as rfile:
						lines = rfile.readlines()
						msg = msg + databasefolder + jpg.replace("./","")
					for i in range(3, len(lines)-1):
						spl = lines[i].split(" ")
						msg = msg + "," + str(int(float(spl[0]) * xratio))#x
						msg = msg + "," + str(int(float(spl[1].replace("\r\n","")) * yratio)) #y
					msg = msg + "\n"
				else:
					msg = msg + databasefolder + jpg.replace("./","") + ",-1\n"

	with open(output_path, 'w') as wfile:
		csvwriter = csv.writer(wfile)
		wfile.write(msg)
