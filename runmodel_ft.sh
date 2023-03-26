#python main.py [model name] [savefolder] [feature point number] [color mode] [mode] [database Pattern] [Test subject number] [Total number of subject]

##LOOCV1 all position, all subject 24 cross-validation
#LOOCV 
for i in {0..23}
do
 python main_ft_VGG16.py VGG16 ./savemodel/LOOCV1_VGG16_1pt 1 gray 2 "./database/LOOCV1_person_%d.csv" $i 24
done

for i in {0..23}
do
 python main_ft_VGG16_do.py VGG16_do ./savemodel/LOOCV1_VGG16_do_1pt 1 gray 2 "./database/LOOCV1_person_%d.csv" $i 24
done

for i in {0..23}
do
 python main_ft_VGG16_bn.py VGG16_bn ./savemodel/LOOCV1_VGG16_bn_1pt 1 gray 2 "./database/LOOCV1_person_%d.csv" $i 24
done

for i in {0..23}
do
 python main_ft_Xception.py Xception_original ./savemodel/LOOCV1_Xception_1pt 1 gray 2 "./database/LOOCV1_person_%d.csv" $i 24
done

for i in {0..23}
do
 python main_ft_Xception_do_2.py Xception ./savemodel/LOOCV1_Xception_do_2_1pt 1 gray 2 "./database/LOOCV1_person_%d.csv" $i 24
done

#All subjects
python main_ft_VGG16.py VGG16 ./savemodel/LOOCV1_VGG16_1pt 1 gray 4 "./database/LOOCV1_person_%d.csv" 0 24
python main_ft_VGG16_bn.py VGG16_bn ./savemodel/LOOCV1_VGG16_bn_1pt 1 gray 4 "./database/LOOCV1_person_%d.csv" 0 24
python main_ft_VGG16_do.py VGG16_do ./savemodel/LOOCV1_VGG16_do_1pt 1 gray 4 "./database/LOOCV1_person_%d.csv" 0 24
python main_ft_Xception.py Xception_original ./savemodel/LOOCV1_Xception_1pt 1 gray 4 "./database/LOOCV1_person_%d.csv" 0 24
python main_ft_Xception_do_2.py Xception ./savemodel/LOOCV1_Xception_do_2_1pt 1 gray 4 "./database/LOOCV1_person_%d.csv" 0 24



##LOOCV2 top position, all subject 6 cross-validation
#LOOCV
for i in {0..5}
do
 python main_ft_VGG16.py VGG16 ./savemodel/LOOCV2_VGG16_1pt 1 gray 2 "./database/LOOCV2_person_%d.csv" $i 6
done

for i in {0..5}
do
 python main_ft_VGG16_do.py VGG16_do ./savemodel/LOOCV2_VGG16_do_1pt 1 gray 2 "./database/LOOCV2_person_%d.csv" $i 6
done

for i in {0..5}
do
 python main_ft_VGG16_bn.py VGG16_bn ./savemodel/LOOCV2_VGG16_bn_1pt 1 gray 2 "./database/LOOCV2_person_%d.csv" $i 6
done

for i in {0..5}
do
 python main_ft_Xception.py Xception_original ./savemodel/LOOCV2_Xception_1pt 1 gray 2 "./database/LOOCV2_person_%d.csv" $i 6
done

for i in {0..5}
do
 python main_ft_Xception_do_2.py Xception ./savemodel/LOOCV2_Xception_do_2_1pt 1 gray 2 "./database/LOOCV2_person_%d.csv" $i 6
done

#All subjects
python main_ft_VGG16.py VGG16 ./savemodel/LOOCV2_VGG16_1pt 1 gray 4 "./database/LOOCV2_person_%d.csv" 0 6
python main_ft_VGG16_bn.py VGG16_bn ./savemodel/LOOCV2_VGG16_bn_1pt 1 gray 4 "./database/LOOCV2_person_%d.csv" 0 6
python main_ft_VGG16_do.py VGG16_do ./savemodel/LOOCV2_VGG16_do_1pt 1 gray 4 "./database/LOOCV2_person_%d.csv" 0 6
python main_ft_Xception.py Xception_original ./savemodel/LOOCV2_Xception_1pt 1 gray 4 "./database/LOOCV2_person_%d.csv" 0 6
python main_ft_Xception_do_2.py Xception ./savemodel/LOOCV2_Xception_do_2_1pt 1 gray 4 "./database/LOOCV2_person_%d.csv" 0 6



##LOOCV3 bottom position, all subject 6 cross-validation
#LOOCV
for i in {0..5}
do
 python main_ft_VGG16.py VGG16 ./savemodel/LOOCV3_VGG16_1pt 1 gray 2 "./database/LOOCV3_person_%d.csv" $i 6
done

for i in {0..5}
do
 python main_ft_VGG16_do.py VGG16_do ./savemodel/LOOCV3_VGG16_do_1pt 1 gray 2 "./database/LOOCV3_person_%d.csv" $i 6
done

for i in {0..5}
do
 python main_ft_VGG16_bn.py VGG16_bn ./savemodel/LOOCV3_VGG16_bn_1pt 1 gray 2 "./database/LOOCV3_person_%d.csv" $i 6
done

for i in {0..5}
do
 python main_ft_Xception.py Xception_original ./savemodel/LOOCV3_Xception_1pt 1 gray 2 "./database/LOOCV3_person_%d.csv" $i 6
done

for i in {0..5}
do
 python main_ft_Xception_do_2.py Xception ./savemodel/LOOCV3_Xception_do_2_1pt 1 gray 2 "./database/LOOCV3_person_%d.csv" $i 6
done

#All subjects
python main_ft_VGG16.py VGG16 ./savemodel/LOOCV3_VGG16_1pt 1 gray 4 "./database/LOOCV3_person_%d.csv" 0 6
python main_ft_VGG16_bn.py VGG16_bn ./savemodel/LOOCV3_VGG16_bn_1pt 1 gray 4 "./database/LOOCV3_person_%d.csv" 0 6
python main_ft_VGG16_do.py VGG16_do ./savemodel/LOOCV3_VGG16_do_1pt 1 gray 4 "./database/LOOCV3_person_%d.csv" 0 6
python main_ft_Xception.py Xception_original ./savemodel/LOOCV3_Xception_1pt 1 gray 4 "./database/LOOCV3_person_%d.csv" 0 6
python main_ft_Xception_do_2.py Xception ./savemodel/LOOCV3_Xception_do_2_1pt 1 gray 4 "./database/LOOCV3_person_%d.csv" 0 6



##LOOCV4 left position, all subject 6 cross-validation
#LOOCV
for i in {0..5}
do
 python main_ft_VGG16.py VGG16 ./savemodel/LOOCV4_VGG16_1pt 1 gray 2 "./database/LOOCV4_person_%d.csv" $i 6
done

for i in {0..5}
do
 python main_ft_VGG16_do.py VGG16_do ./savemodel/LOOCV4_VGG16_do_1pt 1 gray 2 "./database/LOOCV4_person_%d.csv" $i 6
done

for i in {0..5}
do
 python main_ft_VGG16_bn.py VGG16_bn ./savemodel/LOOCV4_VGG16_bn_1pt 1 gray 2 "./database/LOOCV4_person_%d.csv" $i 6
done

for i in {0..5}
do
 python main_ft_Xception.py Xception_original ./savemodel/LOOCV4_Xception_1pt 1 gray 2 "./database/LOOCV4_person_%d.csv" $i 6
done

for i in {0..5}
do
 python main_ft_Xception_do_2.py Xception ./savemodel/LOOCV4_Xception_do_2_1pt 1 gray 2 "./database/LOOCV4_person_%d.csv" $i 6
done

#All subjects
python main_ft_VGG16.py VGG16 ./savemodel/LOOCV4_VGG16_1pt 1 gray 4 "./database/LOOCV4_person_%d.csv" 0 6
python main_ft_VGG16_bn.py VGG16_bn ./savemodel/LOOCV4_VGG16_bn_1pt 1 gray 4 "./database/LOOCV4_person_%d.csv" 0 6
python main_ft_VGG16_do.py VGG16_do ./savemodel/LOOCV4_VGG16_do_1pt 1 gray 4 "./database/LOOCV4_person_%d.csv" 0 6
python main_ft_Xception.py Xception_original ./savemodel/LOOCV4_Xception_1pt 1 gray 4 "./database/LOOCV4_person_%d.csv" 0 6
python main_ft_Xception_do_2.py Xception ./savemodel/LOOCV4_Xception_do_2_1pt 1 gray 4 "./database/LOOCV4_person_%d.csv" 0 6



##LOOCV5 right position, all subject 6 cross-validation
#LOOCV
for i in {0..5}
do
 python main_ft_VGG16.py VGG16 ./savemodel/LOOCV5_VGG16_1pt 1 gray 2 "./database/LOOCV5_person_%d.csv" $i 6
done

for i in {0..5}
do
 python main_ft_VGG16_do.py VGG16_do ./savemodel/LOOCV5_VGG16_do_1pt 1 gray 2 "./database/LOOCV5_person_%d.csv" $i 6
done

for i in {0..5}
do
 python main_ft_VGG16_bn.py VGG16_bn ./savemodel/LOOCV5_VGG16_bn_1pt 1 gray 2 "./database/LOOCV5_person_%d.csv" $i 6
done

for i in {0..5}
do
 python main_ft_Xception.py Xception_original ./savemodel/LOOCV5_Xception_1pt 1 gray 2 "./database/LOOCV5_person_%d.csv" $i 6
done

for i in {0..5}
do
 python main_ft_Xception_do_2.py Xception ./savemodel/LOOCV5_Xception_do_2_1pt 1 gray 2 "./database/LOOCV5_person_%d.csv" $i 6
done

#All subjects
python main_ft_VGG16.py VGG16 ./savemodel/LOOCV5_VGG16_1pt 1 gray 4 "./database/LOOCV5_person_%d.csv" 0 6
python main_ft_VGG16_bn.py VGG16_bn ./savemodel/LOOCV5_VGG16_bn_1pt 1 gray 4 "./database/LOOCV5_person_%d.csv" 0 6
python main_ft_VGG16_do.py VGG16_do ./savemodel/LOOCV5_VGG16_do_1pt 1 gray 4 "./database/LOOCV5_person_%d.csv" 0 6
python main_ft_Xception.py Xception_original ./savemodel/LOOCV5_Xception_1pt 1 gray 4 "./database/LOOCV5_person_%d.csv" 0 6
python main_ft_Xception_do_2.py Xception ./savemodel/LOOCV5_Xception_do_2_1pt 1 gray 4 "./database/LOOCV5_person_%d.csv" 0 6








