#!/bin/bash

datapath=$1
#datapath='/media/koustav/Naihati/Dataset/Tiny_ImageNet/'
#cd datapath
#ls tiny-imagenet-200/train/ > folderlist.txt
#awk 'FNR==NR{a[$1];next}($1 in a){print}' file2 file1 > class_mapping.txt



cd $datapath
mkdir data
mkdir data/train
mkdir data/val
echo "extracting..."
unzip -qq tiny-imagenet-200.zip -d temp/

echo "creating training data..."
for folder in `ls temp/tiny-imagenet-200/train`
	do
		mkdir data/train/$folder/
		mkdir data/val/$folder/
		mv temp/tiny-imagenet-200/train/$folder/images/*.JPEG data/train/$folder/
	done

awk '{printf("%s %s\n",$1,$2)}' temp/tiny-imagenet-200/val/val_annotations.txt > ./validation_data.id

echo "creating validation data..."
while read -r line
do
    arr=($line)
	cp temp/tiny-imagenet-200/val/images/`echo ${arr[0]}` data/val/${arr[1]}/${arr[0]}
done < validation_data.id
echo "Complete"

rm -rf temp
