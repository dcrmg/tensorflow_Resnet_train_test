# coding: utf-8
import os
import sys
import argparse


parser = argparse.ArgumentParser(description='define train and test data')

parser.add_argument('--create_data', default='train', type=str, action='store',
                    help='train or test')
args = parser.parse_args()


database_list = ['data/train_data/berkeley','data/train_data/caffe','data/train_data/concert','data/train_data/eagle']
label_list = [0,1,2,3]

root = os.getcwd()
f=open('train.txt', "w+") if args.create_data == 'train' else open('val.txt',"w+")
# f=open('/val.txt', "a+")

label = 0
for item in database_list:
    data_dir = os.path.join(root,item)
    for dirs in os.listdir(data_dir):
       append_item = os.path.join(data_dir,dirs)+' '+str(label_list[label])+'\n'
       f.writelines(append_item)
    label+=1

f.close()