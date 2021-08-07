# -*- coding: utf-8 -*-
# @Time    : 2021/6/27 10:25
# @Author  : He Ruizhi
# @File    : simple_rnn.py
# @Software: PyCharm

import paddle
import numpy as np
from emb_softmax import get_data_loader
import warnings
warnings.filterwarnings('ignore')


class SimpleRNNModel(paddle.nn.Layer):
    def __init__(self, emb_size, hidden_size):
        super(SimpleRNNModel, self).__init__()
        self.emb = paddle.nn.Embedding(num_embeddings=5149, embedding_dim=emb_size)
        self.simple_rnn = paddle.nn.SimpleRNN(input_size=emb_size, hidden_size=hidden_size)
        self.fc = paddle.nn.Linear(in_features=hidden_size, out_features=2)
        self.softmax = paddle.nn.Softmax()

    def forward(self, x):
        x = self.emb(x)
        # SimpleRNN分别返回所有时刻状态和最后时刻状态，这里只使用最后时刻状态
        _, x = self.simple_rnn(x)
        # 去掉第0维，这么处理与PaddlePaddle的SimpleRNN层返回格式有关
        x = paddle.squeeze(x, axis=0)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def train(model, epochs, train_loader, test_loader):
    # 将模型设置为训练模式
    model.train()
    # 定义优化器
    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader):
            sent = data[0]
            label = data[1]

            # 前向计算输出
            logits = model(sent)
            # 计算损失
            loss = paddle.nn.functional.cross_entropy(logits, label, use_softmax=False)

            if batch_id % 50 == 0:
                acc = paddle.metric.accuracy(logits, label)
                print("epoch: {}, batch_id: {}, loss: {}, acc:{}".format(epoch, batch_id, loss.numpy(), acc.numpy()))

            # 后向传播
            loss.backward()
            # 参数更新
            opt.step()
            # 清除梯度
            opt.clear_grad()

        # 每个epoch数据遍历结束后评估模型
        # 将模型设置为评估模式
        model.eval()
        accuracies = []
        losses = []

        for batch_id, data in enumerate(test_loader):
            sent = data[0]
            label = data[1]

            logits = model(sent)
            loss = paddle.nn.functional.cross_entropy(logits, label, use_softmax=False)
            acc = paddle.metric.accuracy(logits, label)

            accuracies.append(acc.numpy())
            losses.append(loss.numpy())

        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
        print("[validation] accuracy/loss: {}/{}".format(avg_acc, avg_loss))

        # 将模型重新设置为训练模式
        model.train()


if __name__ == '__main__':
    # 设置超参数
    seq_len = 100  # 每条文本的长度
    emb_size = 16  # 词嵌入（word embedding大小）
    hidden_size = 32  # Simple RNN中状态向量维度

    # 加载和处理数据
    train_data_loader = get_data_loader('train', seq_len=seq_len, data_show=1)
    test_data_loader = get_data_loader('test', seq_len=seq_len)

    # 创建模型对象
    model = SimpleRNNModel(emb_size, hidden_size)
    # 查看模型结构信息
    paddle.summary(model, input_size=(None, seq_len), dtypes='int64')

    # 训练模型
    train(model, 3, train_data_loader, test_data_loader)
