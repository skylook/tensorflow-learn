# -*- coding: utf-8 -*-
import input_dataset
import forward_prop
import tensorflow as tf
import os
import numpy as np

max_iter_num = 100000  # 设置参数迭代次数
checkpoint_path = './checkpoint'  # 设置模型参数文件所在路径
event_log_path = './event-log'  # 设置事件文件所在路径，用于周期性存储Summary缓存对象


def train():
    with tf.Graph().as_default():  # 指定当前图为默认graph
        global_step = tf.Variable(initial_value=0,
                                  trainable=False)  # 设置trainable=False,是因为防止训练过程中对global_step变量也进行滑动更新操作
        img_batch, label_batch = input_dataset.preprocess_input_data()  # 输入图像的预处理，包括亮度、对比度、图像翻转等操作
        # img_batch, label_batch = input_dataset.input_data(eval_flag=False)
        logits = forward_prop.network(img_batch)  # 图像信号的前向传播过程
        total_loss = forward_prop.loss(logits, label_batch)  # 计算损失
        one_step_gradient_update = forward_prop.one_step_train(total_loss, global_step)  # 返回一步梯度更新操作
        # 创建一个saver对象，用于保存参数到文件中
        saver = tf.train.Saver(
            var_list=tf.all_variables())  # tf.all_variables return a list of `Variable` objects
        all_summary_obj = tf.summary.merge_all()  # 返回所有summary对象先merge再serialize后的的字符串类型tensor
        initiate_variables = tf.global_variables_initializer()
        # log_device_placement参数可以记录每一个操作使用的设备，这里的操作比较多，就不需要记录了，故设置为False
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            sess.run(initiate_variables)  # 变量初始化
            tf.train.start_queue_runners(sess=sess)  # 启动所有的queuerunners
            Event_writer = tf.summary.FileWriter(logdir=event_log_path, graph=sess.graph)
            for step in range(max_iter_num):
                _, loss_value = sess.run(fetches=[one_step_gradient_update, total_loss])
                assert not np.isnan(loss_value)  # 用于验证当前迭代计算出的loss_value是否合理
                if step % 10 == 0:
                    print('step %d, the loss_value is %.2f' % (step, loss_value))
                if step % 100 == 0:
                    # 添加`Summary`协议缓存到事件文件中，故不能写total_loss变量到事件文件中，因为这里的total_loss为普通的tensor类型
                    all_summaries = sess.run(all_summary_obj)
                    Event_writer.add_summary(summary=all_summaries, global_step=step)
                if step % 1000 == 0 or (step + 1) == max_iter_num:
                    variables_save_path = os.path.join(checkpoint_path, 'model-parameters.bin')  # 路径合并，返回合并后的字符串
                    saver.save(sess, variables_save_path,
                               global_step=step)  # 把所有变量（包括moving average前后的模型参数）保存在variables_save_path路径下


if __name__ == '__main__':
    train()