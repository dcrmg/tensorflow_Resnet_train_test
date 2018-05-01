# tensorflow_Resnet_train_test
Code for training different architectures( DenseNet, ResNet, AlexNet, GoogLeNet, VGG, NiN) on your own dataset + Multi-GPU support + batch and single image testing support

This repository provides an easy-to-use way for training and testing different well-known deep learning architectures on your own datasets.
The code directly load images from disk. Moreover, multi-GPU and transfer learning is also supported, also, you can choose testing images in batch or single.

Based on repository:

https://github.com/arashno/tensorflow_multigpu_imagenet


#Example of usages:

Training:
1. Prepare training data list:
python train_val_datalist_creater.py --create_data train

2. training or Transfer learning:
python train.py
or
python train.py --transfer_mode 1 --architecture resnet --retrain_from ./model

Testing:
1. Prepare testing data list:
python train_val_datalist_creater.py --create_data val

2. testing in batch or single:
python eval.py
or
python eval.py --test_images_path ./data/train_data
