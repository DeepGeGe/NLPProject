# -*- coding: utf-8 -*-
# @Time    : 2021/6/16 22:17
# @Author  : He Ruizhi
# @File    : emb_softmax.py
# @Software: PyCharm

import paddle
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
print(paddle.__version__)  # 2.1.0


class IMDBDataset(paddle.io.Dataset):
    """继承paddle.io.Dataset类，创建自定义数据集类"""
    def __init__(self, sents, labels):
        super(IMDBDataset, self).__init__()
        assert len(sents) == len(labels)
        self.sents = sents
        self.labels = labels

    def __getitem__(self, index):
        data = self.sents[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.sents)


def get_data_loader(mode, seq_len=200, batch_size=128, pad_token='<pad>', data_show=None):
    """ 加载训练/测试数据

    :param mode: 加载数据模式 train/test
    :param seq_len: 文本数据中，每一句话的长度都是不一样的，为了方便后续的神经网络的计算，须通过阶段或填充方式统一输入序列长度
    :param batch_size: 批次大小
    :param pad_token: 当一条文本长度小于seq_len时的填充符号
    :param data_show: 如果不为None，则展示数据集中随机data_show条数据
    :return: paddle.io.DataLoader对象
    """
    imdb_data = paddle.text.Imdb(mode=mode)
    # 获取词表字典
    word_dict = imdb_data.word_idx
    # 将pad_token加入最后一个位置
    word_dict[pad_token] = len(word_dict)
    # 获取pad_token的id
    pad_id = word_dict[pad_token]

    # 将数据处理成同样长度
    def create_padded_dataset(dataset):
        # 处理后的句子
        padded_sents = []
        # 对应标签
        labels = []
        for batch_id, data in enumerate(dataset):
            sent, label = data[0], data[1]
            padded_sent = np.concatenate([sent[:seq_len], [pad_id] * (seq_len - len(sent))])
            padded_sents.append(padded_sent)
            labels.append(label)
        return np.array(padded_sents, dtype='int64'), np.array(labels, dtype='int64')

    # 获取处理后的数据
    sents_padded, labels_padded = create_padded_dataset(imdb_data)
    # 使用自定义paddle.io.Dataset类封装
    dataset_obj = IMDBDataset(sents_padded, labels_padded)
    shuffle = True if mode == 'train' else False
    data_loader = paddle.io.DataLoader(dataset_obj, shuffle=shuffle, batch_size=batch_size, drop_last=True)

    if data_show is not None:
        # 定义ids转word方法
        def ids_to_str(ids):
            words = []
            for k in ids:
                w = list(word_dict)[k]
                words.append(w if isinstance(w, str) else w.decode('utf-8'))
            return ' '.join(words)

        show_ids = random.sample(range(len(imdb_data)), data_show)
        for i in show_ids:
            show_sent = imdb_data.docs[i]
            show_label = imdb_data.labels[i]
            print('the {}-th sentence list id is:{}'.format(i+1, show_sent))
            print('the {}-th sentence list is:{}'.format(i+1, ids_to_str(show_sent)))
            print('the {}-th sentence label id is:{}'.format(i+1, show_label))
            print('--------------------------------------------------------')

    return data_loader


def emb_softmax_classifier_model(emb_size=16, seq_len=200):
    """ 创建emb层+softmax分类器层 模型
    其中num_embeddings=5149为实现查看数据集中的单词数
    out_features=2因为情感分类，输出为两类

    :param emb_size: emb大小
    :param seq_len: 单个seq长度
    :return: paddle.Model对象
    """
    net = paddle.nn.Sequential(
        paddle.nn.Embedding(num_embeddings=5149, embedding_dim=emb_size),
        paddle.nn.Flatten(),
        paddle.nn.Linear(in_features=seq_len * emb_size, out_features=2)
    )
    return paddle.Model(net)


if __name__ == '__main__':
    # 设置超参数
    seq_len = 200  # 每条文本的长度
    emb_size = 16  # 词嵌入（word embedding大小）

    # 加载和处理数据
    train_data_loader = get_data_loader('train', seq_len=seq_len, data_show=1)
    test_data_loader = get_data_loader('test', seq_len=seq_len)

    model = emb_softmax_classifier_model(emb_size=emb_size, seq_len=seq_len)
    # 打印模型结构信息
    model.summary(input_size=(None, seq_len), dtype='int64')
    # 配置模型
    model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
                  loss=paddle.nn.CrossEntropyLoss(use_softmax=True), metrics=paddle.metric.Accuracy())

    model.fit(train_data_loader, epochs=5, verbose=1)
    print('测试结果：', model.evaluate(test_data_loader, verbose=0))
