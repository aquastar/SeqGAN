import model
import numpy as np
import tensorflow as tf
import random
import time
from gen_dataloader import Gen_Data_loader, Likelihood_data_loader
from dis_dataloader import Dis_dataloader
from text_classifier import TextCNN
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import cPickle
from config import BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, SEED, positive_file


#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################

PRE_EPOCH_NUM = 100  # 240
TRAIN_ITER = 1  # generator

##########################################################################################

TOTAL_BATCH = 100  # 800

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = EMB_DIM
dis_filter_sizes = [1, 2,3,4]  # , 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200,200,200]  # , 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.5
dis_l2_reg_lambda = 0.2

# Training parameters
dis_batch_size = 64
dis_num_epochs = 3
dis_alter_epoch = 64  # 50

feat_style = 0  # 0-text, 1-img, 2-txt+img, 3-txt+img+mm
category = 'homicide'
if feat_style == 0:
    positive_file = '../dataset/feat_pool/' + category + '_in_num_w2v.npy'
elif feat_style == 1:
    positive_file = '../dataset/feat_pool/' + category + '_in_num_img.npy'
elif feat_style == 2:
    positive_file = '../dataset/feat_pool/' + category + '_in_num_w2v_img.npy'
elif feat_style == 3:
    positive_file = '../dataset/feat_pool/' + category + '_in_num_w2v_img_mm.npy'

negative_file = 'target_generate/generator_sample.npy'
eval_file = 'target_generate/eval_file.npy'
final_trans_file_seqgan = '../trans_file_seqgan.npy'
final_trans_file_mle = '../trans_file_mle.npy'

generated_num = 10000
denominator_epsilon = 0.01


##############################################################################################


class PoemGen(model.LSTM):
    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer()  # ignore learning rate


def get_trainable_model(num_emb):
    return PoemGen(num_emb, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    #  Generated Samples
    generated_samples = []
    start = time.time()
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    end = time.time()
    # print 'Sample generation time:', (end - start)
    np.save(output_file, generated_samples)

    # with open(output_file, 'w') as fout:
    #     for poem in generated_samples:
    #         buffer = ' '.join([str(x) for x in poem]) + '\n'
    #         # buffer = u''.join([words[x] for x in poem]).encode('utf-8') + '\n'
    #         fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def significance_test(sess, target_lstm, data_loader, output_file):
    loss = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.out_loss, {target_lstm.x: batch})
        loss.extend(list(g_loss))
    with open(output_file, 'w')as fout:
        for item in loss:
            buffer = str(item) + '\n'
            fout.write(buffer)


def pre_train_epoch(sess, trainable_model, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss, g_pred = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Likelihood_data_loader(BATCH_SIZE)
    vocab_size = 5000
    dis_data_loader = Dis_dataloader()

    best_score = 1000
    generator = get_trainable_model(vocab_size)

    # oracle model : target lstm
    # target_params = cPickle.load(open('save/target_params.pkl'))
    # target_lstm = TARGET_LSTM(vocab_size, 64, 32, 32, SEQ_LENGTH, 0, target_params)

    with tf.variable_scope('discriminator'):
        cnn = TextCNN(
            sequence_length=SEQ_LENGTH,
            num_classes=2,
            vocab_size=vocab_size,
            embedding_size=dis_embedding_dim,
            filter_sizes=dis_filter_sizes,
            num_filters=dis_num_filters,
            l2_reg_lambda=dis_l2_reg_lambda)

    cnn_params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
    # Define Discriminator Training procedure
    dis_global_step = tf.Variable(0, name="global_step", trainable=False)
    dis_optimizer = tf.train.AdamOptimizer(1e-4)
    dis_grads_and_vars = dis_optimizer.compute_gradients(cnn.loss, cnn_params, aggregation_method=2)
    dis_train_op = dis_optimizer.apply_gradients(dis_grads_and_vars, global_step=dis_global_step)

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # czq
    # generate real data
    # generate_samples(sess, target_lstm, 64, 10000, positive_file)

    # store real data for next step
    positive_data = np.load(positive_file).tolist()
    gen_data_loader.create_batches(positive_data)

    log = open('log/seq_mle_experiment-log.txt', 'w')
    #  pre-train generator
    print '#########################################################################'
    print 'Start pre-training generator...'
    log.write('pre-training...\n')

    for epoch in xrange(PRE_EPOCH_NUM):
        # print 'pre-train epoch:', epoch
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            # generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            # likelihood_data_loader.create_batches(eval_file)
            # test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            # print 'pre-train epoch ', epoch, 'test_loss ', test_loss
            # buffer = str(epoch) + ' ' + str(test_loss) + '\n'
            buffer = 'pre-trained generator:' + str(epoch) + ' ' + str(loss)
            print(buffer)
            log.write(buffer + '\n')

    # generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
    # likelihood_data_loader.create_batches(eval_file)
    # test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
    # buffer = 'After pre-training:' + ' ' + str(test_loss) + '\n'
    buffer = 'After pre-training:' + ' ' + str(loss)
    print(buffer)
    log.write(buffer + '\n')

    # generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
    # likelihood_data_loader.create_batches(eval_file)
    # significance_test(sess, target_lstm, likelihood_data_loader, 'significance/supervise.txt')

    # test purpose only
    generate_samples(sess, generator, BATCH_SIZE, 100, final_trans_file_mle)

    # exit(0)

    print 'Start pre-training discriminator...'
    for _ in range(dis_alter_epoch):
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)

        #  train discriminator
        dis_x_train, dis_y_train = dis_data_loader.load_train_data(positive_file, negative_file)
        dis_batches = dis_data_loader.batch_iter(
            zip(dis_x_train, dis_y_train), dis_batch_size, dis_num_epochs
        )

        for batch in dis_batches:
            try:
                x_batch, y_batch = zip(*batch)
                feed = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: dis_dropout_keep_prob
                }
                _, step = sess.run([dis_train_op, dis_global_step], feed)
            except Exception as e:
                # print str(e)
                raise

        loss = sess.run(cnn.loss, feed)
        buffer = 'pre-train discriminator' + ' ' + str(loss)
        print buffer
        log.write(buffer + '\n')

    rollout = ROLLOUT(generator, 0.8)
    print('Before GAN')
    print '#########################################################################'
    print 'Start Reinforcement Training Generator...'
    log.write('Reinforcement Training...\n')

    # for tensorboard
    # writer = tf.summary.FileWriter('./tb_logs', graph=tf.get_default_graph())

    for total_batch in range(TOTAL_BATCH):
        print 'progress', total_batch, '/', TOTAL_BATCH
        for it in range(TRAIN_ITER):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, cnn)
            feed = {generator.x: samples, generator.rewards: rewards}
            _, g_loss, pre_loss = sess.run([generator.g_updates, generator.g_loss, generator.pretrain_loss],
                                           feed_dict=feed)
            buffer = 'G-step:' + str(TRAIN_ITER) + ':' + str(g_loss) + '|' + str(pre_loss)
            log.write(buffer + '\n')
            print(buffer)
            # if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
            #     generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            #     likelihood_data_loader.create_batches(eval_file)
            #     # test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            #     # buffer = str(total_batch) + ' ' + str(test_loss) + '\n'
            #     # print 'total_batch: ', total_batch, 'test_loss: ', test_loss
            #     log.write(buffer)

            # if test_loss < best_score:
            #     best_score = test_loss
            #     print 'best score: ', test_loss
            #     significance_test(sess, target_lstm, likelihood_data_loader, 'significance/seqgan.txt')

        rollout.update_params()

        # generate for discriminator
        print('Start training discriminator')
        log.write('training discriminator...\n')

        for _ in range(5):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)

            dis_x_train, dis_y_train = dis_data_loader.load_train_data(positive_file, negative_file)
            dis_batches = dis_data_loader.batch_iter(zip(dis_x_train, dis_y_train), dis_batch_size, 3)

            for batch in dis_batches:
                try:
                    x_batch, y_batch = zip(*batch)
                    feed = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _, step = sess.run([dis_train_op, dis_global_step], feed)
                except ValueError:
                    pass

            loss = sess.run(cnn.loss, feed)
            buffer = 'discriminator' + ' ' + str(loss)
            print buffer
            log.write(buffer + '\n')

    log.close()

    # save the model
    # saver = tf.train.Saver({"gen": generator})
    # saver.save(sess, 'my-model')

    # generate samples
    generate_samples(sess, generator, BATCH_SIZE, 100, final_trans_file_seqgan)


if __name__ == '__main__':
    main()
