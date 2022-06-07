"""Estimators for compressed sensing"""
# pylint: disable = C0301, C0103, C0111, R0914

import copy
import heapq
import os
import math
import tensorflow as tf
import numpy as np
import utils
import scipy.fftpack as fftpack
import pywt

import celebA_model_def
from celebA_utils import save_image


def dct2(image_channel):
    return fftpack.dct(fftpack.dct(image_channel.T, norm='ortho').T, norm='ortho')


def idct2(image_channel):
    return fftpack.idct(fftpack.idct(image_channel.T, norm='ortho').T, norm='ortho')


def vec(channels):
    image = np.zeros((64, 64, 3))
    for i, channel in enumerate(channels):
        image[:, :, i] = channel
    return image.reshape([-1])


def devec(vector):
    image = np.reshape(vector, [64, 64, 3])
    channels = [image[:, :, i] for i in range(3)]
    return channels


def wavelet_basis(path='./src/wavelet_basis.npy'):
    W_ = np.load(path)
    # W_ initially has shape (4096,64,64), i.e. 4096 64x64 images
    # reshape this into 4096x4096, where each row is an image
    # take transpose to make columns images
    W_ = W_.reshape((4096, 4096))
    W = np.zeros((12288, 12288))
    W[0::3, 0::3] = W_
    W[1::3, 1::3] = W_
    W[2::3, 2::3] = W_
    return W


def lasso_dct_estimator(hparams):  #pylint: disable = W0613
    """LASSO with DCT"""
    def estimator(A_val, y_batch_val, hparams):
        # One can prove that taking 2D DCT of each row of A,
        # then solving usual LASSO, and finally taking 2D ICT gives the correct answer.
        A_new = copy.deepcopy(A_val)
        for i in range(A_val.shape[1]):
            A_new[:, i] = vec([dct2(channel) for channel in devec(A_new[:, i])])

        x_hat_batch = []
        for j in range(hparams.batch_size):
            y_val = y_batch_val[j]
            z_hat = utils.solve_lasso(A_new, y_val, hparams)
            x_hat = vec([idct2(channel) for channel in devec(z_hat)]).T
            x_hat = np.maximum(np.minimum(x_hat, 1), -1)
            x_hat_batch.append(x_hat)
        return x_hat_batch
    return estimator


def lasso_wavelet_estimator(hparams):  #pylint: disable = W0613
    """LASSO with Wavelet"""
    def estimator(A_val, y_batch_val, hparams):
        x_hat_batch = []
        W = wavelet_basis()
        WA = np.dot(W, A_val)
        for j in range(hparams.batch_size):
            y_val = y_batch_val[j]
            z_hat = utils.solve_lasso(WA, y_val, hparams)
            x_hat = np.dot(z_hat, W)
            x_hat_max = np.abs(x_hat).max()
            x_hat = x_hat / (1.0 * x_hat_max)
            x_hat_batch.append(x_hat)
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch
    return estimator


def lasso_wavelet_ycbcr_estimator(hparams):  #pylint: disable = W0613
    """LASSO with Wavelet in YCbCr"""

    def estimator(A_val, y_batch_val, hparams):
        x_hat_batch = []

        W = wavelet_basis()
        # U, V = utils.RGB_matrix()
        # V = (V/127.5) - 1.0
        # U = U/127.5
        def convert(W):
            # convert W from YCbCr to RGB
            W_ = W.copy()
            V = np.zeros((12288, 1))
            # R
            V[0::3] = ((255.0/219.0)*(-16.0)) + ((255.0*0.701/112.0)*(-128.0))
            W_[:, 0::3] = (255.0/219.0)*W[:, 0::3] + (0.0)*W[:, 1::3] + (255.0*0.701/112.0)*W[:, 2::3]
            # G
            V[1::3] = ((255.0/219.0)*(-16.0)) - ((0.886*0.114*255.0/(112.0*0.587)) *(-128.0)) - ((255.0*0.701*0.299/(112.0*0.587))*(-128.0))
            W_[:, 1::3] = (255.0/219.0)*W[:, 0::3] - (0.886*0.114*255.0/(112.0*0.587))*W[:, 1::3] - (255.0*0.701*0.299/(112.0*0.587))*W[:, 2::3]
            # B
            V[2::3] = ((255.0/219.0)*(-16.0)) + ((0.886*255.0/(112.0))*(-128.0))
            W_[:, 2::3] = (255.0/219.0)*W[:, 0::3]  + (0.886*255.0/(112.0))*W[:, 1::3] + 0.0*W[:, 2::3]
            return W_, V

        # WU = np.dot(W, U.T)
        WU, V = convert(W)
        WU = WU/127.5
        V = (V/127.5) - 1.0
        WA = np.dot(WU, A_val)
        y_batch_val_temp = y_batch_val - np.dot(V.T, A_val)
        for j in range(hparams.batch_size):
            y_val = y_batch_val_temp[j]
            z_hat = utils.solve_lasso(WA, y_val, hparams)
            x_hat = np.dot(z_hat, WU) + V.ravel()
            x_hat_max = np.abs(x_hat).max()
            x_hat = x_hat / (1.0 * x_hat_max)
            x_hat_batch.append(x_hat)
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch

    return estimator


def dcgan_estimator(hparams):
    # pylint: disable = C0326

    # Get a session
    sess = tf.Session()

    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

    # Create the generator
    z_batch = tf.Variable(tf.random_normal([hparams.batch_size, 100]), name='z_batch')
    x_hat_batch, restore_dict_gen, restore_path_gen = celebA_model_def.dcgan_gen(z_batch, hparams)

    # Create the discriminator
    prob, restore_dict_discrim, restore_path_discrim = celebA_model_def.dcgan_discrim(x_hat_batch, hparams)

    # measure the estimate
    if hparams.measurement_type == 'project':
        y_hat_batch = tf.identity(x_hat_batch, name='y2_batch')
    else:
        measurement_is_sparse = (hparams.measurement_type in ['inpaint', 'superres'])
        y_hat_batch = tf.matmul(x_hat_batch, A, b_is_sparse=measurement_is_sparse, name='y2_batch')

    # define all losses
    m_loss1_batch =  tf.reduce_mean(tf.abs(y_batch - y_hat_batch), 1)
    m_loss2_batch =  tf.reduce_mean((y_batch - y_hat_batch)**2, 1)
    zp_loss_batch =  tf.reduce_sum(z_batch**2, 1)
    d_loss1_batch = -tf.log(prob)
    d_loss2_batch =  tf.log(1-prob)

    # define total loss
    total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                     + hparams.mloss2_weight * m_loss2_batch \
                     + hparams.zprior_weight * zp_loss_batch \
                     + hparams.dloss1_weight * d_loss1_batch \
                     + hparams.dloss2_weight * d_loss2_batch
    total_loss = tf.reduce_mean(total_loss_batch)

    # Compute means for logging
    m_loss1 = tf.reduce_mean(m_loss1_batch)
    m_loss2 = tf.reduce_mean(m_loss2_batch)
    zp_loss = tf.reduce_mean(zp_loss_batch)
    d_loss1 = tf.reduce_mean(d_loss1_batch)
    d_loss2 = tf.reduce_mean(d_loss2_batch)

    # Set up gradient descent
    var_list = [z_batch]
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = utils.get_learning_rate(global_step, hparams)
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        opt = utils.get_optimizer(learning_rate, hparams)
        update_op = opt.minimize(total_loss, var_list=var_list, global_step=global_step, name='update_op')
    opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    restorer_gen = tf.train.Saver(var_list=restore_dict_gen)
    restorer_discrim = tf.train.Saver(var_list=restore_dict_discrim)
    restorer_gen.restore(sess, restore_path_gen)
    restorer_discrim.restore(sess, restore_path_discrim)

    def estimator(A_val, y_batch_val, hparams):
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams)

        if hparams.measurement_type == 'project':
            feed_dict = {y_batch: y_batch_val}
        else:
            feed_dict = {A: A_val, y_batch: y_batch_val}

        for i in range(hparams.num_random_restarts):
            sess.run(opt_reinit_op)
            for j in range(hparams.max_update_iter):
                if hparams.gif and ((j % hparams.gif_iter) == 0):
                    images = sess.run(x_hat_batch, feed_dict=feed_dict)
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)

                _, lr_val, total_loss_val, \
                m_loss1_val, \
                m_loss2_val, \
                zp_loss_val, \
                d_loss1_val, \
                d_loss2_val = sess.run([update_op, learning_rate, total_loss,
                                        m_loss1,
                                        m_loss2,
                                        zp_loss,
                                        d_loss1,
                                        d_loss2], feed_dict=feed_dict)
                logging_format = 'rr {} iter {} lr {} total_loss {} m_loss1 {} m_loss2 {} zp_loss {} d_loss1 {} d_loss2 {}'
                print logging_format.format(i, j, lr_val, total_loss_val,
                                            m_loss1_val,
                                            m_loss2_val,
                                            zp_loss_val,
                                            d_loss1_val,
                                            d_loss2_val)

            x_hat_batch_val, total_loss_batch_val = sess.run([x_hat_batch, total_loss_batch], feed_dict=feed_dict)
            best_keeper.report(x_hat_batch_val, total_loss_batch_val)
        return best_keeper.get_best()

    return estimator


def k_sparse_wavelet_estimator(hparams): #pylint: disable = W0613
    """Best k-sparse wavelet projector"""
    def estimator(A_val, y_batch_val, hparams): #pylint: disable = W0613
        if hparams.measurement_type != 'project':
            raise RuntimeError
        y_batch_val /= np.sqrt(hparams.n_input)
        x_hat_batch = []
        for y_val in y_batch_val:
            y_val_reshaped = np.reshape(y_val, [64, 64, 3])
            x_hat_reshaped = k_sparse_reconstr(y_val_reshaped, hparams.sparsity)
            x_hat_flat = np.reshape(x_hat_reshaped, [-1])
            x_hat_batch.append(x_hat_flat)
        x_hat_batch = np.asarray(x_hat_batch)
        x_hat_batch = np.maximum(np.minimum(x_hat_batch, 1), -1)
        return x_hat_batch
    return estimator


def dcgan_Baysian_estimator(hparams):
    """CSGM + Bayesian Inference, which can be efficiently performed by GD + early stopping"""
    # pylint: disable = C0326

    # Get a session
    sess = tf.Session()

    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

    # Create the generator
    z_batch = tf.Variable(tf.random_normal([hparams.batch_size, 100]), name='z_batch')
    x_hat_batch, restore_dict_gen, restore_path_gen = celebA_model_def.dcgan_gen(z_batch, hparams)
    #########
    restorer_gen = tf.train.Saver(var_list=restore_dict_gen)
    restorer_gen.restore(sess, restore_path_gen) # put in the loop
    constant_list = []
    theta_loss_batch = 0
    for var, var_value in restore_dict_gen.items():
        val = sess.run(var_value)
        constant_list.append(tf.constant(val, tf.float32,shape=val.shape, name=str(var)+'_constant'))
        theta_loss_batch += tf.reduce_mean((var_value-constant_list[-1] )** 2)
    ############

    # Create the discriminator
    # prob, restore_dict_discrim, restore_path_discrim = celebA_model_def.dcgan_discrim(x_hat_batch, hparams)

    # measure the estimate
    if hparams.measurement_type == 'project':
        y_hat_batch = tf.identity(x_hat_batch, name='y2_batch')
    else:
        measurement_is_sparse = (hparams.measurement_type in ['inpaint', 'superres'])
        y_hat_batch = tf.matmul(x_hat_batch, A, b_is_sparse=measurement_is_sparse, name='y2_batch')

    # define all losses
    m_loss1_batch =  tf.reduce_mean(tf.abs(y_batch - y_hat_batch), 1)
    m_loss2_batch = tf.reduce_mean((y_batch - y_hat_batch) ** 2, 1)
    zp_loss_batch = tf.reduce_sum(z_batch ** 2, 1)
    # d_loss1_batch = -tf.log(prob)
    # d_loss2_batch =  tf.log(1-prob)

    # define total loss
    total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                     + hparams.mloss2_weight * m_loss2_batch \
                     + hparams.zprior_weight * zp_loss_batch \
                     + hparams.theta_loss_weight * theta_loss_batch
                     # + hparams.dloss1_weight * d_loss1_batch \
                     # + hparams.dloss2_weight * d_loss2_batch \
    total_loss = tf.reduce_mean(total_loss_batch)

    # Compute means for logging
    m_loss1 = tf.reduce_mean(m_loss1_batch)
    m_loss2 = tf.reduce_mean(m_loss2_batch)
    zp_loss = tf.reduce_mean(zp_loss_batch)
    # d_loss1 = tf.reduce_mean(d_loss1_batch)
    # d_loss2 = tf.reduce_mean(d_loss2_batch)

    # Set up GD for z
    var_list_z = [z_batch]
    global_step_z = tf.Variable(0, trainable=False, name='global_step_z')
    learning_rate_z = utils.get_learning_rate(global_step_z, hparams)
    with tf.variable_scope('GD_z', reuse=False):
        opt_z = utils.get_optimizer(learning_rate_z, hparams)
        update_op_z = opt_z.minimize(total_loss, var_list=var_list_z, global_step=global_step_z, name='update_op_z')
    opt_reinit_op = utils.get_opt_reinit_op(opt_z, var_list_z, global_step_z)

    # Set up GD for theta
    var_list_theta = restore_dict_gen.values()
    global_step_theta = tf.Variable(0, trainable=False, name='global_step_theta')
    learning_rate_theta = tf.constant(hparams.learning_rate_theta)
    with tf.variable_scope('GD_theta', reuse=False):
        opt_theta = tf.train.AdamOptimizer(learning_rate_theta, beta1=0.5)
        update_op_theta = opt_theta.minimize(total_loss, var_list=var_list_theta, global_step=global_step_theta, name='update_op_theta')

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    restorer_gen = tf.train.Saver(var_list=restore_dict_gen)

    # restorer_discrim = tf.train.Saver(var_list=restore_dict_discrim)
    restorer_gen.restore(sess, restore_path_gen) # put in the loop
    # restorer_discrim.restore(sess, restore_path_discrim)

    def estimator(A_val, y_batch_val, hparams):
        """Function that returns the estimated image"""

        best_keeper = utils.BestKeeper(hparams)
        if hparams.measurement_type == 'project':
            feed_dict = {y_batch: y_batch_val}
        else:
            feed_dict = {A: A_val, y_batch: y_batch_val}

        for i in range(hparams.num_random_restarts):
            restorer_gen.restore(sess, restore_path_gen)    # restore theta
            sess.run(opt_reinit_op)     # restore z
            for j in range(hparams.max_update_iter):
                if (j + 1) % hparams.logging_iter == 0:
                    _, lr_val, total_loss_val, \
                    m_loss2_val, \
                    zp_loss_val = sess.run([update_op_z, learning_rate_z, total_loss,
                                            m_loss2,
                                            zp_loss], feed_dict=feed_dict)
                    logging_format = 'Updating z: rr {} iter {} lr {} total_loss {} m_loss2 {} zp_loss {} '
                    print logging_format.format(i, j + 1, lr_val, total_loss_val,
                                                m_loss2_val,
                                                zp_loss_val)
                else:
                    _, lr_val = sess.run([update_op_z, learning_rate_z], feed_dict=feed_dict)

            for j in range(hparams.max_update_iter_theta):
                if (j + 1) % hparams.logging_iter_theta == 0:
                    _, lr_val, total_loss_val, \
                    m_loss2_val, \
                    zp_loss_val = sess.run([update_op_theta, learning_rate_theta, total_loss,
                                                m_loss2,
                                                zp_loss], feed_dict=feed_dict)
                    logging_format = 'Updating theta: rr {} iter {} lr {} total_loss {} m_loss2 {} zp_loss {} '
                    print logging_format.format(i, j + 1, lr_val, total_loss_val,
                                                m_loss2_val,
                                                zp_loss_val)
                else:
                    _, lr_val = sess.run([update_op_theta, learning_rate_theta], feed_dict=feed_dict)

            x_hat_batch_val, total_loss_batch_val = sess.run([x_hat_batch, total_loss_batch], feed_dict=feed_dict)
            best_keeper.report(x_hat_batch_val, total_loss_batch_val)

        return best_keeper.get_best()

    return estimator



def KL_div_batch(p, q, eleNum):
    # p: (mu1, d2) ---- (mu1, diag(d2)I)
    # q: (mu2, sigma2)      ---- (mu2, sigma2*I)
    a = eleNum * tf.log(q[1])
    b = tf.reduce_sum(tf.log(p[1]))
    c = (tf.reduce_sum(p[1]) + tf.reduce_sum((p[0] - q[0]) ** 2)) / q[1]

    return 0.5 * (a - b - eleNum + c)


def dcgan_KL_estimator(hparams):
    sess = tf.Session()
    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

    pz_mu_batch = tf.Variable(0.0 * tf.random_normal([hparams.batch_size, 100]), dtype=tf.float32, name='mu_batch')
    pz_d_batch = tf.Variable(hparams.pzdmodify * math.sqrt(hparams.sigma_z2) * tf.ones([hparams.batch_size, 100]), dtype=tf.float32, name='pz_sigma')
    qz_mu = tf.zeros([hparams.batch_size, 100], dtype=tf.float32, name="qz_mu")
    qz_sigma2 = tf.constant(hparams.sigma_z2, dtype=tf.float32, name="qz_sigma2")

    m_loss2_batch_sampled = 0
    zp_loss_batch_sampled = 0

    x_average = 0
    for _ in range(hparams.sample_num_z):
        z_batch = pz_mu_batch + tf.random_normal([hparams.batch_size, 100]) * tf.abs(pz_d_batch)
        x_hat_batch, restore_dict_gen, restore_path_gen = celebA_model_def.dcgan_gen(z_batch, hparams)
        if hparams.measurement_type == 'project':
            y_hat_batch_sampled = tf.identity(x_hat_batch, name='y2_batch')
        else:
            measurement_is_sparse = (hparams.measurement_type in ['inpaint', 'superres'])
            y_hat_batch = tf.matmul(x_hat_batch, A, b_is_sparse=measurement_is_sparse, name='y2_batch')

        m_loss2_batch = tf.reduce_mean((y_batch - y_hat_batch) ** 2, 1)
        zp_loss_batch = tf.reduce_sum(z_batch ** 2, 1)

        m_loss2_batch_sampled += m_loss2_batch
        zp_loss_batch_sampled += zp_loss_batch
        x_average += x_hat_batch

    x_average = x_average / hparams.sample_num_z
    KL_pz = (pz_mu_batch, tf.abs(pz_d_batch) ** 2)
    KL_qz = (qz_mu, qz_sigma2)
    KL_z_loss_batch = KL_div_batch(KL_pz, KL_qz, eleNum=sess.run(tf.size(pz_mu_batch)))

    restorer_gen = tf.train.Saver(var_list=restore_dict_gen)
    restorer_gen.restore(sess, restore_path_gen)  # put in the loop
    constant_list = []
    theta_loss_batch = 0
    for var, var_value in restore_dict_gen.items():
        val = sess.run(var_value)
        constant_list.append(tf.constant(val, tf.float32, shape=val.shape, name=str(var) + '_constant'))
        theta_loss_batch += tf.reduce_mean((var_value - constant_list[-1]) ** 2)




    # define total loss
    total_loss_batch = hparams.mloss2_weight * m_loss2_batch_sampled # \
    # + hparams.zprior_weight * zp_loss_batch_sampled

    total_loss = tf.reduce_mean(total_loss_batch) / hparams.sample_num_z + hparams.KL_z_weight * KL_z_loss_batch \
                 + hparams.theta_loss_weight * theta_loss_batch

    # Compute means for logging
    m_loss2 = tf.reduce_mean(m_loss2_batch_sampled / hparams.sample_num_z)
    zp_loss = tf.reduce_mean(zp_loss_batch_sampled / hparams.sample_num_z)

    # Set up GD for z
    var_list1 = [pz_mu_batch, pz_d_batch]
    global_step1 = tf.Variable(0, trainable=False, name='global_step1')
    learning_rate1 = utils.get_learning_rate(global_step1, hparams)
    with tf.variable_scope('GD_z', reuse=False):
        opt1 = utils.get_optimizer(learning_rate1, hparams)
        update_op1 = opt1.minimize(total_loss, var_list=var_list1, global_step=global_step1, name='update_op1')
    opt_reinit_op = utils.get_opt_reinit_op(opt1, var_list1, global_step1)

    # Set up GD for theta
    # var_list2 = restore_dict_gen.values()

    var_list2 = restore_dict_gen.values()

    global_step2 = tf.Variable(0, trainable=False, name='global_step2')
    learning_rate2 = tf.constant(0.0001)
    with tf.variable_scope('GD_theta', reuse=False):
        opt2 = tf.train.AdamOptimizer(learning_rate2, beta1=0.5)
        update_op2 = opt2.minimize(total_loss, var_list=var_list2, global_step=global_step2, name='update_op2')

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    restorer_gen = tf.train.Saver(var_list=restore_dict_gen)
    restorer_gen.restore(sess, restore_path_gen)

    def estimator(A_val, y_batch_val, hparams):
        """Function that returns the estimated image"""

        best_keeper = utils.BestKeeper(hparams)

        if hparams.measurement_type == 'project':
            feed_dict = {y_batch: y_batch_val}
        else:
            feed_dict = {A: A_val, y_batch: y_batch_val}

        for i in range(hparams.num_random_restarts):
            restorer_gen.restore(sess, restore_path_gen)
            sess.run(opt_reinit_op)
            for j in range(hparams.max_update_iter):
                if hparams.gif and ((j % hparams.gif_iter) == 0):
                    images = sess.run(x_hat_batch, feed_dict=feed_dict)
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)

                if (j + 1) % hparams.logging_iter == 0:
                    _, \
                    lr_val, \
                    total_loss_val, \
                    m_loss2_val, \
                    zp_loss_val, \
                    KL_z_loss_batch_val = sess.run([update_op1,
                                               learning_rate1,
                                               total_loss,
                                               m_loss2,
                                               zp_loss,
                                               KL_z_loss_batch], feed_dict=feed_dict)
                    logging_format = 'z: rr {} iter {} lr {} total_loss {} m_loss2 {} zp_loss {} KL_z_loss {}'
                    print logging_format.format(i,
                                                j + 1,
                                                lr_val,
                                                total_loss_val,
                                                m_loss2_val,
                                                zp_loss_val,
                                                KL_z_loss_batch_val)
                else:
                    _, lr_val = sess.run([update_op1, learning_rate1], feed_dict=feed_dict)

            for j in range(hparams.max_update_iter_theta):
                if (j + 1) % hparams.logging_iter_theta == 0:
                    _, \
                    lr_val, \
                    total_loss_val, \
                    m_loss2_val, \
                    zp_loss_val, \
                    KL_z_loss_batch_val = sess.run([update_op2,
                                               learning_rate2,
                                               total_loss,
                                               m_loss2,
                                               zp_loss,
                                               KL_z_loss_batch], feed_dict=feed_dict)
                    logging_format = 'z: rr {} iter {} lr {} total_loss {} m_loss2 {} zp_loss {} KL_z_loss {}'
                    print logging_format.format(i,
                                                j + 1,
                                                lr_val,
                                                total_loss_val,
                                                m_loss2_val,
                                                zp_loss_val,
                                                KL_z_loss_batch_val
                                                )

                else:
                    _, lr_val, x_hat_batch_val = sess.run([update_op2, learning_rate2, x_hat_batch],
                                                          feed_dict=feed_dict)


            x_hat_batch_val, total_loss_batch_val = sess.run([x_average, total_loss_batch], feed_dict=feed_dict)
            best_keeper.report(x_hat_batch_val, total_loss_batch_val)


        return best_keeper.get_best()

    return estimator



def get_wavelet(x):
    coefs_list = []
    for i in range(3):
        coefs_list.append(pywt.wavedec2(x[:, :, i], 'db1'))
    return coefs_list


def get_image(coefs_list):
    x = np.zeros((64, 64, 3))
    for i in range(3):
        x[:, :, i] = pywt.waverec2(coefs_list[i], 'db1')
    return x


def get_heap(coefs_list):
    heap = []
    for t, coefs in enumerate(coefs_list):
        for i, a in enumerate(coefs):
            for j, b in enumerate(a):
                for m, c in enumerate(b):
                    try:
                        for n, val in enumerate(c):
                            heapq.heappush(heap, (-abs(val), [t, i, j, m, n, val]))
                    except:
                        val = c
                        heapq.heappush(heap, (-abs(val), [t, i, j, m, val]))
    return heap


def k_sparse_reconstr(x, k):
    coefs_list = get_wavelet(x)
    heap = get_heap(coefs_list)

    y = 0*x
    coefs_list_sparse = get_wavelet(y)
    for i in range(k):
        _, idxs_val = heapq.heappop(heap)
        if len(idxs_val) == 5:
            t, i, j, m, val = idxs_val
            coefs_list_sparse[t][i][j][m] = val
        else:
            t, i, j, m, n, val = idxs_val
            coefs_list_sparse[t][i][j][m][n] = val
    x_sparse = get_image(coefs_list_sparse)
    return x_sparse
