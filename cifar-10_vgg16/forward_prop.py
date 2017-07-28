# -*- coding: utf-8 -*-
import tensorflow as tf
import input_dataset

# 外部引用input_dataset文件中定义的hyperparameters
height = input_dataset.fixed_height
width = input_dataset.fixed_width
train_samples_per_epoch = input_dataset.train_samples_per_epoch
test_samples_per_epoch = input_dataset.test_samples_per_epoch

# 用于描述训练过程的常数
moving_average_decay = 0.9999  # The decay to use for the moving average.
num_epochs_per_decay = 350.0  # 衰减呈阶梯函数，控制衰减周期（阶梯宽度）
learning_rate_decay_factor = 0.1  # 学习率衰减因子
initial_learning_rate = 0.1  # 初始学习率


def variable_on_cpu(name, shape, dtype, initializer):
    with tf.device("/cpu:0"):  # 一个 context manager,用于为新的op指定要使用的硬件
        return tf.get_variable(name=name,
                               shape=shape,
                               initializer=initializer,
                               dtype=dtype)


def variable_on_cpu_with_collection(name, shape, dtype, stddev, wd):
    with tf.device("/cpu:0"):
        weight = tf.get_variable(name=name,
                                 shape=shape,
                                 initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(weight), wd, name='weight_loss')
            tf.add_to_collection(name='losses', value=weight_decay)
        return weight


def losses_summary(total_loss):
    # 通过使用指数衰减，来维护变量的滑动均值。当训练模型时，维护训练参数的滑动均值是有好处的。在测试过程中使用滑动参数比最终训练的参数值本身，
    # 会提高模型的实际性能（准确率）。apply()方法会添加trained variables的shadow copies，并添加操作来维护变量的滑动均值到shadow copies。average
    # 方法可以访问shadow variables，在创建evaluation model时非常有用。
    # 滑动均值是通过指数衰减计算得到的。shadow variable的初始化值和trained variables相同，其更新公式为
    # shadow_variable = decay * shadow_variable + (1 - decay) * variable
    average_op = tf.train.ExponentialMovingAverage(decay=0.9)  # 创建一个新的指数滑动均值对象
    losses = tf.get_collection(key='losses')  # 从字典集合中返回关键字'losses'对应的所有变量，包括交叉熵损失和正则项损失
    # 创建‘shadow variables’,并添加维护滑动均值的操作
    maintain_averages_op = average_op.apply(losses + [total_loss])  # 维护变量的滑动均值，返回一个能够更新shadow variables的操作
    for i in losses + [total_loss]:
        tf.summary.scalar(i.op.name + '_raw', i)  # 保存变量到Summary缓存对象，以便写入到文件中
        tf.summary.scalar(i.op.name,
                          average_op.average(i))  # average() returns the shadow variable for a given variable.
    return maintain_averages_op  # 返回损失变量的更新操作


def one_step_train(total_loss, step):
    batch_count = int(train_samples_per_epoch / input_dataset.batch_size)  # 求训练块的个数
    decay_step = batch_count * num_epochs_per_decay  # 每经过decay_step步训练，衰减lr
    lr = tf.train.exponential_decay(learning_rate=initial_learning_rate,
                                    global_step=step,
                                    decay_steps=decay_step,
                                    decay_rate=learning_rate_decay_factor,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    losses_movingaverage_op = losses_summary(total_loss)
    # tf.control_dependencies是一个context manager,控制节点执行顺序，先执行control_inputs中的操作，再执行context中的操作
    with tf.control_dependencies(control_inputs=[losses_movingaverage_op]):
        trainer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        gradient_pairs = trainer.compute_gradients(loss=total_loss)  # 返回计算出的（gradient, variable） pairs
    gradient_update = trainer.apply_gradients(grads_and_vars=gradient_pairs, global_step=step)  # 返回一步梯度更新操作
    # num_updates参数用于动态调整衰减率，真实的decay_rate =min(decay, (1 + num_updates) / (10 + num_updates)
    variables_average_op = tf.train.ExponentialMovingAverage(decay=moving_average_decay, num_updates=step)
    # tf.trainable_variables() 方法返回所有`trainable=True`的变量，列表结构
    maintain_variable_average_op = variables_average_op.apply(var_list=tf.trainable_variables())  # 返回模型参数变量的滑动更新操作
    with tf.control_dependencies(control_inputs=[gradient_update, maintain_variable_average_op]):
        gradient_update_optimizor = tf.no_op()  # Does nothing. Only useful as a placeholder for control edges
    return gradient_update_optimizor


def network(images):
    # 这一部分主要调用几个常见函数，在上一篇博客‘TensorFlow实现卷积神经网络’中有详细介绍，这里就不再赘述～
    with tf.variable_scope(name_or_scope='conv1') as scope:
        weight = variable_on_cpu_with_collection(name='weight',
                                                 shape=(5, 5, 3, 64),
                                                 dtype=tf.float32,
                                                 stddev=0.05,
                                                 wd=0.0)
        bias = variable_on_cpu(name='bias', shape=(64), dtype=tf.float32,
                               initializer=tf.constant_initializer(value=0.0))
        conv1_in = tf.nn.conv2d(input=images, filter=weight, strides=(1, 1, 1, 1), padding='SAME')
        conv1_in = tf.nn.bias_add(value=conv1_in, bias=bias)
        conv1_out = tf.nn.relu(conv1_in)

    pool1 = tf.nn.max_pool(value=conv1_out, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')

    norm1 = tf.nn.lrn(input=pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    with tf.variable_scope(name_or_scope='conv2') as scope:
        weight = variable_on_cpu_with_collection(name='weight',
                                                 shape=(5, 5, 64, 64),
                                                 dtype=tf.float32,
                                                 stddev=0.05,
                                                 wd=0.0)
        bias = variable_on_cpu(name='bias', shape=(64), dtype=tf.float32,
                               initializer=tf.constant_initializer(value=0.1))
        conv2_in = tf.nn.conv2d(norm1, weight, strides=(1, 1, 1, 1), padding='SAME')
        conv2_in = tf.nn.bias_add(conv2_in, bias)
        conv2_out = tf.nn.relu(conv2_in)

    norm2 = tf.nn.lrn(input=conv2_out, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    pool2 = tf.nn.max_pool(value=norm2, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')
    # input tensor of shape `[batch, in_height, in_width, in_channels]
    reshaped_pool2 = tf.reshape(tensor=pool2, shape=(-1, 6 * 6 * 64))

    with tf.variable_scope(name_or_scope='fully_connected_layer1') as scope:
        weight = variable_on_cpu_with_collection(name='weight',
                                                 shape=(6 * 6 * 64, 384),
                                                 dtype=tf.float32,
                                                 stddev=0.04,
                                                 wd=0.004)
        bias = variable_on_cpu(name='bias', shape=(384), dtype=tf.float32,
                               initializer=tf.constant_initializer(value=0.1))
        fc1_in = tf.matmul(reshaped_pool2, weight) + bias
        fc1_out = tf.nn.relu(fc1_in)

    with tf.variable_scope(name_or_scope='fully_connected_layer2') as scope:
        weight = variable_on_cpu_with_collection(name='weight',
                                                 shape=(384, 192),
                                                 dtype=tf.float32,
                                                 stddev=0.04,
                                                 wd=0.004)
        bias = variable_on_cpu(name='bias', shape=(192), dtype=tf.float32,
                               initializer=tf.constant_initializer(value=0.1))
        fc2_in = tf.matmul(fc1_out, weight) + bias
        fc2_out = tf.nn.relu(fc2_in)

    with tf.variable_scope(name_or_scope='softmax_layer') as scope:
        weight = variable_on_cpu_with_collection(name='weight',
                                                 shape=(192, 10),
                                                 dtype=tf.float32,
                                                 stddev=1 / 192,
                                                 wd=0.0)
        bias = variable_on_cpu(name='bias', shape=(10), dtype=tf.float32,
                               initializer=tf.constant_initializer(value=0.0))
        classifier_in = tf.matmul(fc2_out, weight) + bias
        classifier_out = tf.nn.softmax(classifier_in)
    return classifier_out


def loss(logits, labels):
    labels = tf.cast(x=labels, dtype=tf.int32)  # 强制类型转换，使符合sparse_softmax_cross_entropy_with_logits输入参数格式要求
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                        name='likelihood_loss')
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_loss')  # 对batch_size长度的向量取平均
    tf.add_to_collection(name='losses', value=cross_entropy_loss)  # 把张量cross_entropy_loss添加到字典集合中key='losses'的子集中
    return tf.add_n(inputs=tf.get_collection(key='losses'), name='total_loss')  # 返回字典集合中key='losses'的子集中元素之和