"""Evaluating a trained model on the test data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os

import numpy as np
import tensorflow as tf
import argparse
import arch
import data_loader
import sys
from PIL import Image
import imghdr


def evaluate(args):

  # Building the graph
  with tf.Graph().as_default() as g, tf.device('/cpu:0'):
      # Get images and labels for CIFAR-10.
      if args.save_predictions is None:
          images, labels = data_loader.read_inputs(False, args)
      else:
          images, labels, urls = data_loader.read_inputs(False, args)
      # Performing computations on a GPU
      with tf.device('/gpu:0'):
          # Build a Graph that computes the logits predictions from the
          # inference model.
          logits = arch.get_model(images, 0.0, False, args)

          # Calculate predictions accuracies top-1 and top-n
          top_1_op = tf.nn.in_top_k(logits, labels, 1)
          top_n_op = tf.nn.in_top_k(logits, labels, args.top_n)

          if args.save_predictions is not None:
            topn = tf.nn.top_k(tf.nn.softmax(logits), args.top_n)
            topnind= topn.indices
            topnval= topn.values

          saver = tf.train.Saver(tf.global_variables())

          # Build the summary operation based on the TF collection of Summaries.
          summary_op = tf.summary.merge_all()

          summary_writer = tf.summary.FileWriter(args.log_dir, g)

      with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
          sess.run(tf.global_variables_initializer())
          sess.run(tf.local_variables_initializer())

          ckpt = tf.train.get_checkpoint_state(args.log_dir)

          # Load the latest model
          if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
              saver.restore(sess, ckpt.model_checkpoint_path)

          else:
              return
          # Start the queue runners.
          coord = tf.train.Coordinator()

          threads = tf.train.start_queue_runners(sess=sess, coord=coord)
          true_predictions_count = 0  # Counts the number of correct predictions
          true_topn_predictions_count = 0
          all_count = 0
          step = 0
          predictions_format_str = ('%d,%s,%d,%s,%s\n')
          batch_format_str = ('Batch Number: %d, Top-1 Hit: %d, Top-'+str(args.top_n)+' Hit: %d, Top-1 Accuracy: %.3f, Top-'+str(args.top_n)+' Accuracy: %.3f')

          if args.save_predictions is not None:
              out_file = open(args.save_predictions,'w')
          while step < args.num_batches and not coord.should_stop():
              try:
                  if args.save_predictions is None:
                      top1_predictions, topn_predictions, bac = sess.run([top_1_op, top_n_op, logits])
                      # print('%%%%%%%%%%%%%%%%%%')
                      # print(bac)
                      # for i in logits:
                      #     print(i)

                      # print(logits.eval())
                      # abc = sess.run(logits)
                  else:
                      top1_predictions, topn_predictions, urls_values, label_values, topnguesses, topnconf = sess.run(
                          [top_1_op, top_n_op, urls, labels, topnind, topnval])
                      for i in xrange(0, urls_values.shape[0]):
                          out_file.write(
                              predictions_format_str % (step * args.batch_size + i + 1, urls_values[i], label_values[i],
                                                        '[' + ', '.join('%d' % item for item in topnguesses[i]) + ']',
                                                        '[' + ', '.join('%.4f' % item for item in topnconf[i]) + ']'))
                          out_file.flush()
              except Exception,e:
                  coord.request_stop(e)

              true_predictions_count += np.sum(top1_predictions)
              true_topn_predictions_count += np.sum(topn_predictions)
              all_count+= top1_predictions.shape[0]
              print(batch_format_str%(step, true_predictions_count, true_topn_predictions_count, true_predictions_count / all_count, true_topn_predictions_count / all_count))
              sys.stdout.flush()
              step += 1

          if args.save_predictions is not None:
              out_file.close()
 
          summary = tf.Summary()
          summary.ParseFromString(sess.run(summary_op))
          coord.request_stop()
          coord.join(threads)

def get_sess(args):
  with tf.Graph().as_default() as g, tf.device('/gpu:0'):

      input = tf.placeholder(dtype=tf.float32, shape=(None,args.load_size[0] ,args.load_size[1],args.num_channels))
      logits = arch.get_model(input, 0.0, False,args)

      saver = tf.train.Saver(tf.global_variables())
      sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

      ckpt = tf.train.get_checkpoint_state('./model')

      if ckpt and ckpt.model_checkpoint_path:
          try:
              saver.restore(sess, ckpt.model_checkpoint_path)
          except Exception,e:
              print(e)
              print('Load Model ERROR ERROR!!!!!!!!!!!!!!!!')
              sys.exit()
      else:
          print('model does not exist!! ,The program is going to quit!')
          sys.exit()

      return sess, logits, input

def standardization(img,width=400,height=400,channel=3):
    num_compare = width*height*channel
    img_arr = np.array(img)
    img_std = (img_arr - np.mean(img_arr)) / max(np.std(img_arr), 1 / math.sqrt(num_compare))
    return img_std

def test(args,image_folder='./img'):

    sess, logits, input = get_sess(args)

    for root, dirs, files in os.walk(image_folder):
        for file in files:
            item = os.path.join(root, file)
            try:
                item = unicode(item, 'utf-8')
            except:
                continue

            out = 0

            try:
                if imghdr.what(os.path.join(root, file)) not in {'jpeg','jpg','rgb','gif','tif','bmp','png'}:
                    print('invalid image: {}'.format(os.path.join(root, file)))
                    continue
                img = Image.open(item)
                img = img.resize(args.load_size, Image.ANTIALIAS)
                std_img = standardization(img)
                imgs = [std_img]

                eva_list = sess.run([logits], feed_dict={input: imgs})

                out = np.argmax(eva_list)

            except Exception, e:
                print(e)
                print('evaluate ERROR!!')

            finally:
                print(out)


def main():
  parser = argparse.ArgumentParser(description='Process Command-line Arguments')
  parser.add_argument('--load_size', nargs= 2, default= [500,500], type= int, action= 'store', help= 'The width and height of images for loading from disk')
  parser.add_argument('--crop_size', nargs= 2, default= [500,500], type= int, action= 'store', help= 'The width and height of images after random cropping')
  parser.add_argument('--batch_size', default= 32, type= int, action= 'store', help= 'The testing batch size')
  parser.add_argument('--num_classes', default= 4, type= int, action= 'store', help= 'The number of classes')
  parser.add_argument('--num_channels', default= 3, type= int, action= 'store', help= 'The number of channels in input images')
  parser.add_argument('--num_batches' , default= 1000 , type= int, action= 'store', help= 'The number of batches of data')
  parser.add_argument('--path_prefix' , default='', action= 'store', help= 'The prefix address for images')
  parser.add_argument('--delimiter' , default=' ', action = 'store', help= 'Delimiter for the input files')
  parser.add_argument('--data_info'   , default= 'val.txt', action= 'store', help= 'File containing the addresses and labels of testing images')
  parser.add_argument('--num_threads', default= 1, type= int, action= 'store', help= 'The number of threads for loading data')
  parser.add_argument('--architecture', default= 'resnet', help='The DNN architecture')
  parser.add_argument('--depth', default= 50, type= int, help= 'The depth of ResNet architecture')
  parser.add_argument('--log_dir', default= './model', action= 'store', help='Path for saving Tensorboard info and checkpoints')
  parser.add_argument('--save_predictions', default= None, action= 'store', help= 'Save top-5 predictions of the networks along with their confidence in the specified file')
  parser.add_argument('--top_n', default= 5, type= int, action= 'store', help= 'Specify the top-N accuracy')
  parser.add_argument('--eval_model', default= True, type= bool, action= 'store', help= 'evaluate acc of  model')
  parser.add_argument('--test_images_path', default= './data/train_data', type= str, action= 'store', help= 'test images')

  args = parser.parse_args()
  args.num_samples = sum(1 for line in open(args.data_info))
  if args.num_batches==-1:
    if(args.num_samples%args.batch_size==0):
      args.num_batches= int(args.num_samples/args.batch_size)
    else:
      args.num_batches= int(args.num_samples/args.batch_size)+1

  print(args)

  if args.eval_model:
    evaluate(args)
  if args.test_images_path!=None:
    test(args,image_folder = args.test_images_path)


if __name__ == '__main__':
  main()
