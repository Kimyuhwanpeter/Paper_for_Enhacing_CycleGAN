# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from random import shuffle
from absl import app
from absl import flags
from collections import Counter
from EAM_model import *

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import imageio
import datetime
import sys

flags.DEFINE_string("A_image_txt", "D:/[1]DB/[1]second_paper_DB/[1]First_fold/_MORPH_MegaAge_16_43_fullDB/[2]MegaAge_16_29_and_Morph_30_43/MegaAge_test_16_29.txt", "A image txt directory")

flags.DEFINE_string("A_image_path", "D:/[1]DB/[1]second_paper_DB/[1]First_fold/_MORPH_MegaAge_16_43_fullDB/[1]FullDB/testA/", "A image directory")

flags.DEFINE_integer("Number_A_image", 5052, "Number of A images")

flags.DEFINE_string("B_image_txt", "D:/[1]DB/[1]second_paper_DB/[1]First_fold/_MORPH_MegaAge_16_43_fullDB/[2]MegaAge_16_29_and_Morph_30_43/Morph_test_30_43.txt", "B image txt directory")

flags.DEFINE_string("B_image_path", "D:/[1]DB/[1]second_paper_DB/[1]First_fold/_MORPH_MegaAge_16_43_fullDB/[1]FullDB/testB/", "B image directory")

flags.DEFINE_integer("Number_B_image", 5052, "Number of B images")

flags.DEFINE_bool("pre_checkpoint", True, "Fine tune or continue the train")

flags.DEFINE_string("pre_checkpoint_path", "C:/Users/Yuhwan/Desktop/1249", "Fine tuning or continue the train")

flags.DEFINE_string('sample_dir', 'D:/tensorflow2.0(New_generator_CycleGAN)/sample_images', 'Sample generated images during training')

flags.DEFINE_string("save_checkpoint", "D:/tensorflow2.0(New_generator_CycleGAN)/checkpoint", "Save checkpoint")

flags.DEFINE_float("L1_lambda", 10.0, "Additional cycle loss weight")

flags.DEFINE_integer("epoch", 200, "Training epoch")

flags.DEFINE_integer('epoch_decay', 100, 'Learning rate decay')

flags.DEFINE_integer("options", 64, "Defalut filter size")

flags.DEFINE_integer("batch_size", 2, "Traing batch")

flags.DEFINE_integer("image_size", 256, "Image size")

flags.DEFINE_integer("load_size", 286, "Before cropped image")

flags.DEFINE_integer("num_classes", 14, "Number of classes")

flags.DEFINE_integer("ch", 3, "Image channel")

flags.DEFINE_string('graphs', 'D:/tensorflow2.0(New_generator_CycleGAN)/graphs/', 'Directory of loss graphs')

flags.DEFINE_float('lr', 2e-4, 'learning rate')

flags.DEFINE_integer('A_first_class', 16, 'First class of A data')

flags.DEFINE_integer('B_first_class', 30, 'First class of B data')

###############################################################################################################
flags.DEFINE_bool('train', False, 'True or False')

flags.DEFINE_string('A_test_img', 'D:/[1]DB/[1]second_paper_DB/[1]First_fold/_MORPH_MegaAge_16_43_fullDB/[1]FullDB/testA/', 'Test image path')

flags.DEFINE_string('A_test_txt', 'D:/[1]DB/[1]second_paper_DB/[1]First_fold/_MORPH_MegaAge_16_43_fullDB/[3]MegaAge_30_43_and_Morph_16_29/MegaAge_test_30_43.txt', 'Test text path')

flags.DEFINE_integer('A_n_test', 5279, 'Number of A test images')

flags.DEFINE_string('A_test_output', 'C:/Users/Yuhwan/Pictures/ssd', 'Generated image path')

flags.DEFINE_string('B_test_img', 'D:/[1]DB/[1]second_paper_DB/[1]First_fold/_MORPH_MegaAge_16_43_fullDB/[1]FullDB/testB/', 'Test image path')

flags.DEFINE_string('B_test_txt', 'D:/[1]DB/[1]second_paper_DB/[1]First_fold/_MORPH_MegaAge_16_43_fullDB/[3]MegaAge_30_43_and_Morph_16_29/Morph_test_16_29.txt', 'Test text path')

flags.DEFINE_integer('B_n_test', 5279, 'Number of A test images')

flags.DEFINE_string('B_test_output', 'C:/Users/Yuhwan/Pictures/ssd', 'Generated image path')

flags.DEFINE_string('test_dir', 'B2A', 'A2B or B2A')
###############################################################################################################

FLAGS = flags.FLAGS
FLAGS(sys.argv)

N_A = FLAGS.Number_A_image
N_B = FLAGS.Number_B_image
lr = FLAGS.lr
epoch = FLAGS.epoch
epoch_decay = FLAGS.epoch_decay

len_dataset = min(N_A, N_B)
G_lr_scheduler = LinearDecay(lr, epoch * len_dataset // FLAGS.batch_size, epoch_decay * len_dataset // FLAGS.batch_size)
D_lr_scheduler = LinearDecay(lr, epoch * len_dataset // FLAGS.batch_size, epoch_decay * len_dataset // FLAGS.batch_size)
generator_optimizer = tf.keras.optimizers.Adam(G_lr_scheduler, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(G_lr_scheduler, beta_1=0.5)

def abs_criterion(input, target):
    return tf.reduce_mean(tf.abs(input - target))

def mae_criterion(input, target):
    return tf.reduce_mean((input - target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels, logits))

def age_preserving_loss(linear_one_hot, fake_Alogits, fake_Blogits):

    fake_Alogits = tf.keras.layers.Flatten()(fake_Alogits)
    fake_Blogits = tf.keras.layers.Flatten()(fake_Blogits)

    fake_Alogits = tf.nn.softmax(fake_Alogits[:,0:FLAGS.num_classes], 1) + 0.0001
    fake_Blogits = tf.nn.softmax(fake_Blogits[:,0:FLAGS.num_classes], 1) + 0.0001    

    linear_one_hot = tf.nn.softmax(linear_one_hot) + 0.0001

    a2b_loss = tf.keras.losses.KLDivergence()(linear_one_hot, fake_Blogits)
    b2a_loss = tf.keras.losses.KLDivergence()(linear_one_hot, fake_Alogits)
    
    return a2b_loss, b2a_loss

def age_preserving_loss_for_generator(A_images, B_images, fake_A, fake_B):
    
    fake_A = tf.nn.softmax(fake_A)
    fake_B = tf.nn.softmax(fake_B)
    
    ground_A = tf.nn.softmax(A_images)
    ground_B = tf.nn.softmax(B_images)

    a2b_loss = tf.keras.losses.KLDivergence()(ground_A, fake_B)
    b2a_loss = tf.keras.losses.KLDivergence()(ground_B, fake_A)

    return a2b_loss, b2a_loss

def _func(name, age):

    h = tf.random.uniform([1], 1e-2, (FLAGS.image_size + 30)-FLAGS.image_size)
    h = tf.cast(tf.math.ceil(h[0]), tf.int32)
    w = tf.random.uniform([1], 1e-2, (FLAGS.image_size + 30)-FLAGS.image_size)
    w = tf.cast(tf.math.ceil(w[0]), tf.int32)

    A_image_string = tf.io.read_file(name[0])
    A_image_decoded = tf.image.decode_jpeg(A_image_string, channels=3)
    A_image_decoded = tf.image.resize(A_image_decoded, [FLAGS.image_size + 30, FLAGS.image_size + 30])
    A_image_decoded = A_image_decoded[h:h+FLAGS.image_size, w:w+FLAGS.image_size, :]
    A_image_gray = tf.image.rgb_to_grayscale(A_image_decoded)
    A_image_decoded = tf.image.convert_image_dtype(A_image_decoded, tf.float32) / 127.5 - 1.    
    A_image_gray = tf.image.convert_image_dtype(A_image_gray, tf.float32) / 127.5 -1.

    B_image_string = tf.io.read_file(name[1])
    B_image_decoded = tf.image.decode_jpeg(B_image_string, channels=3)
    B_image_decoded = tf.image.resize(B_image_decoded, [FLAGS.image_size + 30, FLAGS.image_size + 30])
    B_image_decoded = B_image_decoded[h:h+FLAGS.image_size, w:w+FLAGS.image_size, :]
    B_image_gray = tf.image.rgb_to_grayscale(B_image_decoded)
    B_image_decoded = tf.image.convert_image_dtype(B_image_decoded, tf.float32) / 127.5 - 1.    
    B_image_gray = tf.image.convert_image_dtype(B_image_gray, tf.float32) / 127.5 - 1.

    if tf.random.uniform(()) > 0.5:
        A_image_decoded = tf.image.flip_left_right(A_image_decoded)
        A_image_gray = tf.image.flip_left_right(A_image_gray)
        B_image_decoded = tf.image.flip_left_right(B_image_decoded)
        B_image_gray = tf.image.flip_left_right(B_image_gray)


    A_age = age[0] - FLAGS.A_first_class        # 결국에는 A와 B의 원핫은 동일하기 때문에 이것 하나만 써줘도 상관없다.
    linear_one_hot = tf.one_hot(A_age, FLAGS.num_classes)
    linear_one_hot = tf.cast(linear_one_hot, tf.float32)

    A_real_age = age[0]
    B_real_age = age[1]

    image_label_default_1 = (tf.where( tf.zeros((FLAGS.image_size, FLAGS.image_size, A_age), tf.float32)[:,:,:] == 0., -1., 1. ))
    image_label_default_2 = (tf.where( tf.zeros((FLAGS.image_size, FLAGS.image_size, FLAGS.num_classes-A_age-1), tf.float32)[:,:,:] == 0., -1., 1. ))
    A_bili_one_hot = tf.concat([image_label_default_1, A_image_gray, image_label_default_2], 2)
    B_bili_one_hot = tf.concat([image_label_default_1, B_image_gray, image_label_default_2], 2)

    image_label_default_3 = (tf.where( tf.zeros((FLAGS.image_size, FLAGS.image_size, FLAGS.num_classes-1), tf.float32)[:,:,:] == 0., -1., 1. ))

    A_bili_one_hot_1 = tf.concat([A_image_gray,image_label_default_3], 2)
    B_bili_one_hot_1 = tf.concat([B_image_gray,image_label_default_3], 2)

    A_bili_one_hot_2 = tf.concat([image_label_default_3, A_image_gray], 2)
    B_bili_one_hot_2 = tf.concat([image_label_default_3, B_image_gray], 2)

    a = tf.cond(A_age == 0, lambda: A_bili_one_hot_1, lambda: A_bili_one_hot_2)
    b = tf.cond(A_age == 0, lambda: B_bili_one_hot_1, lambda: B_bili_one_hot_2)

    A_result = tf.cond(A_age > 0 and A_age < FLAGS.num_classes - 1, lambda: A_bili_one_hot, lambda: a)     # A batch labelimages
    B_result = tf.cond(A_age > 0 and A_age < FLAGS.num_classes - 1, lambda: B_bili_one_hot, lambda: b)     # B batch labelimages
    
    return A_image_decoded, B_image_decoded, linear_one_hot, A_result, B_result, A_real_age, B_real_age

def _func_test(filename):
    
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = tf.image.resize(image_decoded, [FLAGS.image_size, FLAGS.image_size])
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32) / 127.5 - 1.

    return image_decoded, filename

@tf.function
def train(A_images, B_images, A_batch_labels, B_batch_labels, linear_one_hot,
            generator_A2B, generator_B2A, discriminator_A, discriminator_B, 
            discriminator_A_2, discriminator_B_2):
    with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
        fake_B = generator_A2B(A_images, training=True)
        fake_A_ = generator_B2A(fake_B, training=True)
        fake_A = generator_B2A(B_images, training=True)
        fake_B_ = generator_A2B(fake_A, training=True)

        DA_fake = discriminator_A(fake_A, training=True)
        DB_fake = discriminator_B(fake_B, training=True)
        DA_fake_2 = discriminator_A_2(fake_A, training=True)
        DB_fake_2 = discriminator_B_2(fake_B, training=True)

        # Pixel loss for only generator
        concat_A_data = tf.concat([A_images, A_batch_labels], 3)
        concat_B_data = tf.concat([B_images, B_batch_labels], 3)
        concat_B_fake = tf.concat([fake_B, A_batch_labels], 3)
        concat_A_fake = tf.concat([fake_A, B_batch_labels], 3)
        perceptual_loss_a2b = tf.reduce_mean(tf.abs(concat_A_data - concat_B_fake))
        perceptual_loss_b2a = tf.reduce_mean(tf.abs(concat_B_data - concat_A_fake))
       
        # Age preserving loss for generator
        preserve_age_generator_a2b, preserve_age_generator_b2a = age_preserving_loss_for_generator(concat_A_data, concat_B_data, concat_A_fake, concat_B_fake)

        g_a2b_loss = mae_criterion(DB_fake, tf.ones_like(DB_fake)) \
                    + mae_criterion(DB_fake_2, tf.ones_like(DB_fake_2)) \
                    + (FLAGS.L1_lambda * abs_criterion(A_images, fake_A_)) \
                    + (FLAGS.L1_lambda * abs_criterion(B_images, fake_B_)) \
                    + perceptual_loss_a2b + preserve_age_generator_a2b
        g_b2a_loss = mae_criterion(DA_fake, tf.ones_like(DA_fake)) \
                    + mae_criterion(DA_fake_2, tf.ones_like(DA_fake_2)) \
                    + (FLAGS.L1_lambda * abs_criterion(A_images, fake_A_)) \
                    + (FLAGS.L1_lambda * abs_criterion(B_images, fake_B_)) \
                    + perceptual_loss_b2a + preserve_age_generator_b2a
        g_loss = mae_criterion(DB_fake, tf.ones_like(DB_fake)) \
                    + mae_criterion(DA_fake, tf.ones_like(DA_fake)) \
                    + mae_criterion(DB_fake_2, tf.ones_like(DB_fake_2)) \
                    + mae_criterion(DA_fake_2, tf.ones_like(DA_fake_2)) \
                    + (FLAGS.L1_lambda * abs_criterion(A_images, fake_A_)) \
                    + (FLAGS.L1_lambda * abs_criterion(B_images, fake_B_)) \
                    + 10.0 * (perceptual_loss_a2b + perceptual_loss_b2a) \
                    + preserve_age_generator_a2b + preserve_age_generator_b2a

        DA_real = discriminator_A(A_images, training=True)
        DB_real = discriminator_B(B_images, training=True)
        DA_real_2 = discriminator_A_2(A_images, training=True)
        DB_real_2 = discriminator_B_2(B_images, training=True)

        # Age preserving loss for discriminator
        preserv_age_D_a2b, preserv_age_D_b2a = age_preserving_loss(linear_one_hot,
                                                DA_real_2, DB_real_2)

        preserv_age_D_a2b_2, preserv_age_D_b2a_2 = age_preserving_loss(linear_one_hot,
                                                DA_fake_2, DB_fake_2)

        disc_A_loss = (mae_criterion(DA_real, tf.ones_like(DA_real)) \
                    + mae_criterion(DA_fake, tf.zeros_like(DA_fake))) / 2
        disc_B_loss = (mae_criterion(DB_real, tf.ones_like(DB_real)) \
                    + mae_criterion(DB_fake, tf.zeros_like(DB_fake))) / 2
        disc_A_loss_2 = (mae_criterion(DA_real_2, tf.zeros_like(DA_real_2)) \
                    + mae_criterion(DA_fake_2, tf.ones_like(DA_fake_2))) / 2
        disc_B_loss_2 = (mae_criterion(DB_real_2, tf.zeros_like(DB_real_2)) \
                    + mae_criterion(DB_fake_2, tf.ones_like(DB_fake_2))) / 2
        
        d_loss = disc_A_loss + disc_B_loss + disc_A_loss_2 + disc_B_loss_2 + preserv_age_D_a2b + preserv_age_D_b2a + preserv_age_D_a2b_2 + preserv_age_D_b2a_2
    
    generator_gradients = gen_tape.gradient(g_loss, generator_A2B.trainable_variables + generator_B2A.trainable_variables)

    discriminator_gradients = disc_tape.gradient(d_loss, discriminator_A.trainable_variables + discriminator_B.trainable_variables \
                                                    + discriminator_A_2.trainable_variables + discriminator_B_2.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator_A2B.trainable_variables + generator_B2A.trainable_variables))
    
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator_A.trainable_variables + discriminator_B.trainable_variables  \
                                                + discriminator_A_2.trainable_variables + discriminator_B_2.trainable_variables))

    return g_loss, d_loss

def main(argv=None):
    
    discriminator_A = ConvDiscriminator_v2(input_shape=(FLAGS.image_size, FLAGS.image_size, FLAGS.ch))
    discriminator_B = ConvDiscriminator_v2(input_shape=(FLAGS.image_size, FLAGS.image_size, FLAGS.ch))
    discriminator_A_2 = ConvDiscriminator_v2(input_shape=(FLAGS.image_size, FLAGS.image_size, FLAGS.ch))
    discriminator_B_2 = ConvDiscriminator_v2(input_shape=(FLAGS.image_size, FLAGS.image_size, FLAGS.ch))
    generator_A2B = ResnetGenerator_v2(input_shape=(FLAGS.image_size, FLAGS.image_size, FLAGS.ch))
    generator_B2A = ResnetGenerator_v2(input_shape=(FLAGS.image_size, FLAGS.image_size, FLAGS.ch))
    
    generator_A2B.summary()
    generator_B2A.summary()
    discriminator_B.summary()

    if FLAGS.pre_checkpoint is True:
        # activation map을 보기위해서는 우선 filter값을 확인해야함!!
        ckpt = tf.train.Checkpoint(generator_A2B=generator_A2B,
                           generator_B2A=generator_B2A,
                           discriminator_A=discriminator_A,
                           discriminator_B=discriminator_B,
                           discriminator_A_2=discriminator_A_2,
                           discriminator_B_2=discriminator_B_2,
                           generator_optimizer=generator_optimizer,
                           discriminator_optimizer=discriminator_optimizer)

        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')
            

    if FLAGS.train == True:
        dataA_name = np.loadtxt(FLAGS.A_image_txt, dtype='<U100', skiprows=0, usecols=0)
        dataA_name = [FLAGS.A_image_path + dataA_name_ for dataA_name_ in dataA_name]
        dataA_label = np.loadtxt(FLAGS.A_image_txt, dtype=np.float32, skiprows=0, usecols=1)
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = FLAGS.graphs + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        count = 0
        batch_idx = min(len(dataA_name), len(dataB_name)) // FLAGS.batch_size
        for epoch_ in range(FLAGS.epoch):
            A_image_buf = np.array(dataA_name)
            A_label_buf = np.array(dataA_label).astype(np.int)
            C = Counter(A_label_buf)
            previous = 0
            for j in range(FLAGS.num_classes):
                A_age = j + FLAGS.A_first_class
                thres = C[A_age]
    
                if j != 0:
                    previous += C[A_age - 1]
                    np.random.shuffle(A_image_buf[previous:thres+previous])
                    np.random.shuffle(A_label_buf[previous:thres+previous])
                else:
                    np.random.shuffle(A_image_buf[0:thres])
                    np.random.shuffle(A_label_buf[0:thres])
    
            dataB_name = np.loadtxt(FLAGS.B_image_txt, dtype='<U100', skiprows=0, usecols=0)
            dataB_name = [FLAGS.B_image_path + dataB_name_ for dataB_name_ in dataB_name]
            dataB_label = np.loadtxt(FLAGS.B_image_txt, dtype=np.float32, skiprows=0, usecols=1)
    
            B_image_buf = np.array(dataB_name)
            B_label_buf = np.array(dataB_label).astype(np.int)
            C = Counter(B_label_buf)
            previous = 0
            for j in range(FLAGS.num_classes):
                B_age = j + FLAGS.B_first_class
                thres = C[B_age]
    
                if j != 0:
                    previous += C[B_age - 1]
                    np.random.shuffle(B_image_buf[previous:thres+previous])
                    np.random.shuffle(B_label_buf[previous:thres+previous])
                else:
                    np.random.shuffle(B_image_buf[0:thres])
                    np.random.shuffle(B_label_buf[0:thres])
    
            A = list(zip(A_image_buf, A_label_buf))
            B = list(zip(B_image_buf, B_label_buf))
    
            C = list(zip(A,B))
            shuffle(C)
            A,B = zip(*C)
    
            dataA_names, dataA_labels = zip(*A)
            dataB_names, dataB_labels = zip(*B)
    
            data_names = list(zip(dataA_names, dataB_names))
            data_labels = list(zip(dataA_labels, dataB_labels))
    
            data_generator = tf.data.Dataset.from_tensor_slices((data_names, data_labels))
            data_generator = data_generator.shuffle( min(len(dataA_names), len(dataB_names)) )
            data_generator = data_generator.map(_func)
            data_generator = data_generator.batch(FLAGS.batch_size)
            data_generator = data_generator.prefetch(tf.data.experimental.AUTOTUNE)

            it = iter(data_generator)
            for step in range(batch_idx):
                A_batch_images, B_batch_images, linear_one_hot, A_batch_labelimage, B_batch_labelimage, A_real_age, B_real_age = next(it)

                g_loss, d_loss = train(A_batch_images, B_batch_images, A_batch_labelimage, B_batch_labelimage, linear_one_hot,
                                    generator_A2B, generator_B2A, discriminator_A, discriminator_B,
                                    discriminator_A_2, discriminator_B_2)
                
                with train_summary_writer.as_default():
                    tf.summary.scalar(u'G loss', g_loss, step=count)
                    tf.summary.scalar(u'D loss', d_loss, step=count)
    
                #g_loss, d_loss = train_step(A_batch_images, B_batch_images, generator_A2B, generator_B2A, 
                #                        discriminator_A, discriminator_B, generator_optimizer, discriminator_optimizer)
                print((u"Epoch:[{}] [{}/{}], G_loss:{}, D_loss:{}, lr:{}".format(epoch_ + 1, step, batch_idx, g_loss, d_loss, G_lr_scheduler.current_learning_rate)))
    
                if count % 500 == 0:
                    fake_B_image = generator_A2B(A_batch_images, training=False)
                    fake_A_image = generator_B2A(B_batch_images, training=False)
        
                    plt.imsave(u"{}/{}_fake_B_{}epochs_{}steps.jpg".format(FLAGS.sample_dir,A_real_age[0],epoch_ + 1,count), fake_B_image[0] * 0.5 + 0.5)
                    plt.imsave(u"{}/{}_fake_B_{}epochs_{}steps.jpg".format(FLAGS.sample_dir,A_real_age[1],epoch_ + 1,count), fake_B_image[1] * 0.5 + 0.5)        

                    plt.imsave(u"{}/{}_real_A_{}epochs_{}steps.jpg".format(FLAGS.sample_dir,A_real_age[0],epoch_ + 1,count), A_batch_images[0] * 0.5 + 0.5)
                    plt.imsave(u"{}/{}_real_A_{}epochs_{}steps.jpg".format(FLAGS.sample_dir,A_real_age[1],epoch_ + 1,count), A_batch_images[1] * 0.5 + 0.5)

                    plt.imsave(u"{}/{}_fake_A_{}epochs_{}steps.jpg".format(FLAGS.sample_dir,B_real_age[0],epoch_ + 1,count), fake_A_image[0] * 0.5 + 0.5)
                    plt.imsave(u"{}/{}_fake_A_{}epochs_{}steps.jpg".format(FLAGS.sample_dir,B_real_age[1],epoch_ + 1,count), fake_A_image[1] * 0.5 + 0.5)

                    plt.imsave(u"{}/{}_real_B_{}epochs_{}steps.jpg".format(FLAGS.sample_dir,B_real_age[0],epoch_ + 1,count), B_batch_images[0] * 0.5 + 0.5)
                    plt.imsave(u"{}/{}_real_B_{}epochs_{}steps.jpg".format(FLAGS.sample_dir,B_real_age[1],epoch_ + 1,count), B_batch_images[1] * 0.5 + 0.5)

                    #plt.imshow(fake_A_image[0] * 0.5 + 0.5)
                    #plt.show()
    
                if count % 5000 == 0:
                    model_dir = FLAGS.save_checkpoint
                    folder_name = int(count/5000)
                    folder_neme_str = '%s/%s' % (model_dir, folder_name)
                    if not os.path.isdir(folder_neme_str):
                        print(u"Make {} folder to save checkpoint".format(folder_name))
                        os.makedirs(folder_neme_str)
                    checkpoint = tf.train.Checkpoint(generator_A2B=generator_A2B,
                                                    generator_B2A=generator_B2A,
                                                    discriminator_A=discriminator_A,
                                                    discriminator_B=discriminator_B,
                                                    discriminator_A_2=discriminator_A_2,
                                                    discriminator_B_2=discriminator_B_2,
                                                    generator_optimizer=generator_optimizer,
                                                    discriminator_optimizer=discriminator_optimizer)
                    checkpoint_dir = folder_neme_str + "/" + u"New_generator_CycleGAN_model_{}_steps.ckpt".format(count + 1)
                    checkpoint.save(checkpoint_dir)
    
    
                count += 1
######################################################################################################################################
    else:
        if FLAGS.test_dir == 'A2B':
            print('=================')
            print('Start to A2B.....')
            print('=================')

            name = np.loadtxt(FLAGS.A_test_txt, dtype='<U100', skiprows=0, usecols=0)
            name = [FLAGS.A_test_img + name_ for name_ in name]
    
            data_generator = tf.data.Dataset.from_tensor_slices(name)
            data_generator = data_generator.map(_func_test)
            data_generator = data_generator.batch(1)
            data_generator = data_generator.prefetch(tf.data.experimental.AUTOTUNE)

            summary_writer = tf.summary.create_file_writer('C:/Users/Yuhwan/Pictures/graphs')

            it = iter(data_generator)
            for i in range(len(name)):
                image, filename = next(it)
                fake_B = generator_A2B(image, training=False)

                #if i == 0:
                #    with summary_writer.as_default():
                #        tf.summary.image('layer output channel_0', tf.expand_dims(fake_B[:, :, :, 0], axis=3), step=i, max_outputs=1)
                #        tf.summary.image('layer output channel_1', tf.expand_dims(fake_B[:, :, :, 1], axis=3), step=i, max_outputs=1)
                #        tf.summary.image('layer output channel_2', tf.expand_dims(fake_B[:, :, :, 2], axis=3), step=i, max_outputs=1)

                plt.imsave(u"{}/{}".format(FLAGS.A_test_output,name[i].split('/')[7]), fake_B[0]*0.5 + 0.5)
                if i % 1000 == 0:
                    print(u'Generated {} image(s)'.format(i + 1))

        else:
            name = np.loadtxt(FLAGS.B_test_txt, dtype='<U100', skiprows=0, usecols=0)
            name = [FLAGS.B_test_img + name_ for name_ in name]

            #name = np.loadtxt('C:/Users/Yuhwan/Pictures/testB.txt', dtype='<U100', skiprows=0, usecols=0)

            data_generator = tf.data.Dataset.from_tensor_slices(name)
            data_generator = data_generator.map(_func_test)
            data_generator = data_generator.batch(1)
            data_generator = data_generator.prefetch(tf.data.experimental.AUTOTUNE)

            #summary_writer = tf.summary.create_file_writer('C:/Users/Yuhwan/Pictures/graphs')

            print('=================')
            print('Start to B2A.....')
            print('=================')
            it = iter(data_generator)
            for i in range(FLAGS.B_n_test):
                images, _ = next(it)
                fake_A = generator_B2A(images, training=False)

                #real_A_logits = discriminator_A(images, training=False)
                #fake_A_logits = discriminator_A(fake_A, training=False)

                #shape = fake_A_logits.get_shape()[-1]

                #fake_A_logits = fake_A_logits[0].numpy()
                #fake_A_logits = np.array(fake_A_logits).astype(np.float32)
                #img = 0
                #for j in range(shape):

                #    img += fake_A_logits[:,:,j]

                #img /= shape
                #img = img[:,:,np.newaxis]
                #img = tf.image.resize(img, [256,256]) 
                #img = img.numpy()
                #img = np.squeeze(img, 2)

                #plt.imshow(img, cmap=plt.cm.jet)
                #plt.savefig('C:/Users/Yuhwan/Pictures/discriminator_5.jpg')
                #plt.show()
                

                plt.imsave(u"{}/{}".format(FLAGS.B_test_output,name[i].split('/')[7]), fake_A[0] * 0.5 + 0.5)
                if i % 1000 == 0:
                    print(u'generated {} image(s)'.format(i + 1))


if __name__ == '__main__':
    app.run(main)