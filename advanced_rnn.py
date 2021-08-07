# -*- coding: utf-8 -*-
# @Time    : 2021/7/6 18:55
# @Author  : He Ruizhi
# @File    : advanced_rnn.py
# @Software: PyCharm

import paddle
from emb_softmax import get_data_loader
import warnings
warnings.filterwarnings('ignore')


class AdvancedLSTM(paddle.nn.Layer):
    def __init__(self, embedding_dim, hidden_size):
        super(AdvancedLSTM, self).__init__()
        self.emb = paddle.nn.Embedding(num_embeddings=5149, embedding_dim=embedding_dim)
        self.lstm = paddle.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                                   direction='bidirectional', num_layers=2)
        self.flatten = paddle.nn.Flatten()
        self.fc = paddle.nn.Linear(in_features=hidden_size*2, out_features=2)
        self.softmax = paddle.nn.Softmax()

    def forward(self, x):
        x = self.emb(x)
        # LSTM层分别返回所有时刻状态和最后时刻h与c状态，这里只使用最后时刻的h
        _, (x, _) = self.lstm(x)
        # 获取双层双向LSTM最后一层的输出状态
        x = paddle.slice(x, axes=[0], starts=[2], ends=[4])
        # 调整x的数据排列，这么处理与PaddlePaddle的LSTM层返回格式有关
        x = paddle.transpose(x, [1, 0, 2])
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    seq_len = 200
    emb_size = 16
    hidden_size = 32

    train_data_loader = get_data_loader('train', seq_len=seq_len, data_show=1)
    test_data_loader = get_data_loader('test', seq_len=seq_len)

    model = paddle.Model(AdvancedLSTM(16, 32))
    model.summary(input_size=(None, seq_len), dtype='int64')
    model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters()),
                  loss=paddle.nn.CrossEntropyLoss(use_softmax=False),
                  metrics=paddle.metric.Accuracy())
    model.fit(train_data_loader, epochs=5, verbose=1, eval_data=test_data_loader)
