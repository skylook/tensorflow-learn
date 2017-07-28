# -*- coding: utf-8 -*-
import tensorflow as tf
import input_dataset
import forward_prop
import train
import math
import numpy as np


def eval_once(summary_op, summary_writer, saver, predict_true_or_false):
    with tf.Session() as sess:
        # 从checkpoint文件中返回checkpointstate模板
        checkpoint_proto = tf.train.get_checkpoint_state(checkpoint_dir=train.checkpoint_path)
        if checkpoint_proto and checkpoint_proto.model_checkpoint_path:
            saver.restore(sess, checkpoint_proto.model_checkpoint_path)  # 恢复模型变量到当前session中
        else:
            print('checkpoint file not found!')
            return
        # 启动很多线程，并把coordinator传递给每一个线程
        coord = tf.train.Coordinator()  # 返回一个coordinator类对象，这个类实现了一个简单的机制，可以用来coordinate很多线程的结束
        try:
            threads = []  # 使用coord统一管理所有线程
            for queue_runner in tf.get_collection(key=tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(queue_runner.create_threads(sess, coord=coord, daemon=True, start=True))
            # 计算测试数据块的个数,并向上取整
            test_batch_num = math.ceil(input_dataset.test_samples_per_epoch / input_dataset.batch_size)
            iter_num = 0
            true_test_num = 0
            # 这里使用取整后的测试数据块个数，来计算测试样例的总数目，理论上这样算测试样例总数会偏大啊，暂时还未理解？？？
            total_test_num = test_batch_num * input_dataset.batch_size

            while iter_num < test_batch_num and not coord.should_stop():
                result_judge = sess.run([predict_true_or_false])
                true_test_num += np.sum(result_judge)
                iter_num += 1
            precision = true_test_num / total_test_num
            print("The test precision is %.3f" % precision)
        except:
            coord.request_stop()
        coord.request_stop()
        coord.join(threads)


def evaluate():
    with tf.Graph().as_default() as g:
        img_batch, labels = input_dataset.input_data(eval_flag=True)  # 读入测试数据集
        logits = forward_prop.network(img_batch)  # 使用moving average操作前的模型参数，计算模型输出值
        # 判断targets是否在前k个predictions里面，当k=1时等价于常规的计算正确率的方法，sess.run(predict_true_or_false)会执行符号计算
        predict_true_or_false = tf.nn.in_top_k(predictions=logits, targets=labels, k=1)
        # 恢复moving average操作后的模型参数
        moving_average_op = tf.train.ExponentialMovingAverage(decay=forward_prop.moving_average_decay)
        # 返回要恢复的names到Variables的映射，也即一个map映射。如果一个变量有moving average,就使用moving average变量名作为the restore
        # name, 否则就使用变量名
        variables_to_restore = moving_average_op.variables_to_restore()
        saver = tf.train.Saver(var_list=variables_to_restore)

        summary_op = tf.merge_all_summaries()  # 创建序列化后的summary对象
        # 创建一个event file,用于之后写summary对象到logdir目录下的文件中
        summary_writer = tf.train.SummaryWriter(logdir='./event-log-test', graph=g)
        eval_once(summary_op, summary_writer, saver, predict_true_or_false)