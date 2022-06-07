"""Estimators for compressed sensing"""
# pylint: disable = C0301, C0103, C0111, R0914

from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy as np
import tensorflow as tf
import mnist_model_def
from mnist_utils import save_image
import utils
import math

def lasso_estimator(hparams):  # pylint: disable = W0613
    """LASSO estimator"""
    def estimator(A_val, y_batch_val, hparams):
        x_hat_batch = []
        for i in range(hparams.batch_size):
            y_val = y_batch_val[i]
            x_hat = utils.solve_lasso(A_val, y_val, hparams)
            x_hat = np.maximum(np.minimum(x_hat, 1), 0)
            x_hat_batch.append(x_hat)
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch
    return estimator


def omp_estimator(hparams):
    """OMP estimator"""
    omp_est = OrthogonalMatchingPursuit(n_nonzero_coefs=hparams.omp_k)
    def estimator(A_val, y_batch_val, hparams):
        x_hat_batch = []
        for i in range(hparams.batch_size):
            y_val = y_batch_val[i]
            omp_est.fit(A_val.T, y_val.reshape(hparams.num_measurements))
            x_hat = omp_est.coef_
            x_hat = np.reshape(x_hat, [-1])
            x_hat = np.maximum(np.minimum(x_hat, 1), 0)
            x_hat_batch.append(x_hat)
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch
    return estimator


def vae_estimator(hparams):

    # Get a session
    sess = tf.Session()

    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

    # Create the generator
    # TODO: Move z_batch definition here
    z_batch, x_hat_batch, restore_path, restore_dict = mnist_model_def.vae_gen(hparams)

    # measure the estimate
    if hparams.measurement_type == 'project':
        y_hat_batch = tf.identity(x_hat_batch, name='y_hat_batch')
    else:
        y_hat_batch = tf.matmul(x_hat_batch, A, name='y_hat_batch')

    # define all losses
    m_loss1_batch = tf.reduce_mean(tf.abs(y_batch - y_hat_batch), 1)
    m_loss2_batch = tf.reduce_mean((y_batch - y_hat_batch)**2, 1)
    zp_loss_batch = tf.reduce_sum(z_batch**2, 1)

    # define total loss
    total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                     + hparams.mloss2_weight * m_loss2_batch \
                     + hparams.zprior_weight * zp_loss_batch
    total_loss = tf.reduce_mean(total_loss_batch)

    # Compute means for logging
    m_loss1 = tf.reduce_mean(m_loss1_batch)
    m_loss2 = tf.reduce_mean(m_loss2_batch)
    zp_loss = tf.reduce_mean(zp_loss_batch)

    # Set up gradient descent
    var_list = [z_batch]
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = utils.get_learning_rate(global_step, hparams)
    opt = utils.get_optimizer(learning_rate, hparams)
    update_op = opt.minimize(total_loss, var_list=var_list, global_step=global_step, name='update_op')
    opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

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
                _, lr_val, total_loss_val, \
                m_loss1_val, \
                m_loss2_val, \
                zp_loss_val = sess.run([update_op, learning_rate, total_loss,
                                        m_loss1,
                                        m_loss2,
                                        zp_loss], feed_dict=feed_dict)
                logging_format = 'rr {} iter {} lr {} total_loss {} m_loss1 {} m_loss2 {} zp_loss {}'
                print logging_format.format(i, j, lr_val, total_loss_val,
                                            m_loss1_val,
                                            m_loss2_val,
                                            zp_loss_val)

                if hparams.gif and ((j % hparams.gif_iter) == 0):
                    images = sess.run(x_hat_batch, feed_dict=feed_dict)
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)

            x_hat_batch_val, total_loss_batch_val = sess.run([x_hat_batch, total_loss_batch], feed_dict=feed_dict)
            best_keeper.report(x_hat_batch_val, total_loss_batch_val)
        return best_keeper.get_best()

    return estimator

def mnist_Baysian_estimator(hparams):
    """CSGM + Bayesian Inference, which can be efficiently performed by GD + early stopping"""
    # pylint: disable = C0326

    # Get a session
    sess = tf.Session()

    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

    # Create the generator
    z_batch, x_hat_batch, restore_path_gen, restore_dict_gen = mnist_model_def.vae_gen(hparams)
    #########
    restorer_gen = tf.train.Saver(var_list=restore_dict_gen)
    restorer_gen.restore(sess, restore_path_gen)
    constant_list = []
    theta_loss_batch = 0
    for var, var_value in restore_dict_gen.items():
        val = sess.run(var_value)
        constant_list.append(tf.constant(val, tf.float32,shape=val.shape, name=str(var)+'_constant'))
        theta_loss_batch += tf.reduce_mean((var_value-constant_list[-1])** 2)
    ############

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

    # define total loss
    total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                     + hparams.mloss2_weight * m_loss2_batch \
                     + hparams.zprior_weight * zp_loss_batch 
                     
    total_loss = tf.reduce_mean(total_loss_batch) + hparams.theta_loss_weight * theta_loss_batch

    # Compute means for logging
    # m_loss1 = tf.reduce_mean(m_loss1_batch)
    m_loss2 = tf.reduce_mean(m_loss2_batch)
    zp_loss = tf.reduce_mean(zp_loss_batch)


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
    restorer_gen.restore(sess, restore_path_gen) # put in the loop


    def estimator(A_val, y_batch_val, hparams):
        """Function that returns the estimated image"""

        best_keeper = utils.BestKeeper(hparams)
        if hparams.measurement_type == 'project':
            feed_dict = {y_batch: y_batch_val}
        else:
            feed_dict = {A: A_val, y_batch: y_batch_val}

        for i in range(hparams.num_random_restarts):
            sess.run(opt_reinit_op)     # restore z
            restorer_gen.restore(sess, restore_path_gen)
            for j in range(hparams.max_update_iter):
                if (j + 1) % hparams.logging_iter == 0:
                    _, lr_val, total_loss_val, \
                    m_loss2_val, \
                    zp_loss_val, theta_loss_val = sess.run([update_op_z, learning_rate_z, total_loss,
                                            m_loss2,
                                            zp_loss, theta_loss_batch], feed_dict=feed_dict)
                    logging_format = 'Updating z: rr {} iter {} lr {} total_loss {} m_loss2 {} zp_loss {} theta_loss {}'
                    print logging_format.format(i, j + 1, lr_val, total_loss_val,
                                                m_loss2_val,
                                                zp_loss_val, theta_loss_val)
                else:
                    _, lr_val = sess.run([update_op_z, learning_rate_z], feed_dict=feed_dict)

            for j in range(hparams.max_update_iter_theta):
                if (j + 1) % hparams.logging_iter_theta == 0:
                    _, lr_val, total_loss_val, \
                    m_loss2_val, \
                    zp_loss_val, theta_loss_val = sess.run([update_op_theta, learning_rate_theta, total_loss,
                                                m_loss2,
                                                zp_loss, theta_loss_batch], feed_dict=feed_dict)
                    logging_format = 'Updating theta: rr {} iter {} lr {} total_loss {} m_loss2 {} zp_loss {} theta_loss {}'
                    print logging_format.format(i, j + 1, lr_val, total_loss_val,
                                                m_loss2_val,
                                                zp_loss_val, theta_loss_val)
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
    b = tf.reduce_sum(tf.log(1e-10+p[1]))
    c = (tf.reduce_sum(p[1]) + tf.reduce_sum((p[0] - q[0]) ** 2)) / q[1]

    return 0.5 * (a - b - eleNum + c)


def mnist_KL_estimator(hparams):
    sess = tf.Session()
    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

    pz_mu_batch = tf.Variable(0.0 * tf.random_normal([hparams.batch_size, 20]), dtype=tf.float32, name='mu_batch')
    pz_d_batch = tf.Variable(hparams.pzdmodify * math.sqrt(hparams.sigma_z2) * tf.ones([hparams.batch_size, 20]), dtype=tf.float32, name='pz_sigma')
    qz_mu = tf.zeros([hparams.batch_size, 20], dtype=tf.float32, name="qz_mu")
    qz_sigma2 = tf.constant(hparams.sigma_z2, dtype=tf.float32, name="qz_sigma2")

    m_loss2_batch_sampled = 0
    zp_loss_batch_sampled = 0

    x_average = 0
    for _ in range(hparams.sample_num_z):
        z_batch = pz_mu_batch + tf.random_normal([hparams.batch_size, 20]) * tf.abs(pz_d_batch)
        _, x_hat_batch, restore_path_gen, restore_dict_gen = mnist_model_def.vae_gen(hparams, z_batch)
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
    learning_rate2 = tf.constant(hparams.learning_rate_theta)
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
            sess.run(opt_reinit_op)
            restorer_gen.restore(sess, restore_path_gen)
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
                    theta_loss_val, \
                    KL_z_loss_batch_val = sess.run([update_op1,
                                               learning_rate1,
                                               total_loss,
                                               m_loss2,
                                               zp_loss,
                                               theta_loss_batch,
                                               KL_z_loss_batch], feed_dict=feed_dict)
                    logging_format = 'z: rr {} iter {} lr {} total_loss {} m_loss2 {} zp_loss {} theta_loss {} KL_z_loss {}'
                    print logging_format.format(i,
                                                j + 1,
                                                lr_val,
                                                total_loss_val,
                                                m_loss2_val,
                                                zp_loss_val,
                                                theta_loss_val, 
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
                    theta_loss_val, \
                    KL_z_loss_batch_val = sess.run([update_op2,
                                               learning_rate2,
                                               total_loss,
                                               m_loss2,
                                               zp_loss,
                                               theta_loss_batch,
                                               KL_z_loss_batch], feed_dict=feed_dict)
                    logging_format = 'theta: rr {} iter {} lr {} total_loss {} m_loss2 {} zp_loss {} theta_loss {} KL_z_loss {}'
                    print logging_format.format(i,
                                                j + 1,
                                                lr_val,
                                                total_loss_val,
                                                m_loss2_val,
                                                zp_loss_val,
                                                theta_loss_val, 
                                                KL_z_loss_batch_val
                                                )

                else:
                    _, lr_val, x_hat_batch_val = sess.run([update_op2, learning_rate2, x_hat_batch],
                                                          feed_dict=feed_dict)


            x_hat_batch_val, total_loss_batch_val = sess.run([x_average, total_loss_batch], feed_dict=feed_dict)
            best_keeper.report(x_hat_batch_val, total_loss_batch_val)


        return best_keeper.get_best()

    return estimator


def learned_estimator(hparams):

    sess = tf.Session()
    y_batch, x_hat_batch, restore_dict = mnist_model_def.end_to_end(hparams)
    restore_path = utils.get_A_restore_path(hparams)

    # Intialize and restore model parameters
    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

    def estimator(A_val, y_batch_val, hparams):  # pylint: disable = W0613
        """Function that returns the estimated image"""
        x_hat_batch_val = sess.run(x_hat_batch, feed_dict={y_batch: y_batch_val})
        return x_hat_batch_val

    return estimator
