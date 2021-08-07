# -*- coding: utf-8 -*-
# @Time    : 2021/8/1 19:16
# @Author  : He Ruizhi
# @File    : machine_translation.py
# @Software: PyCharm

import paddle
import paddle.nn.functional as F
import re
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
print(paddle.__version__)  # 2.1.0

# 设置训练句子最大长度，用于筛选数据集中的部分数据
MAX_LEN = 12


def create_dataset(file_path):
    """
    构建机器翻译训练数据集

    :param file_path: 训练数据路径
    :return:
    train_en_sents：由数字ID组成的英文句子
    train_cn_sents：由数字ID组成的中文句子
    train_cn_label_sents：由数字ID组成的中文词汇标签
    en_vocab：英文词表
    cn_vocab：中文词表
    """
    with open(file_path, 'rt', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    # 设置正则匹配模板，用于从英文句子中提取单词
    words_re = re.compile(r'\w+')

    # 将训练数据文件中的中文和英文句子全部提取出来
    pairs = []
    for line in lines:
        en_sent, cn_sent, _ = line.split('\t')
        pairs.append((words_re.findall(en_sent.lower())+[en_sent[-1]], list(cn_sent)))

    # 从原始训练数据中筛选出一部分数据用来训练模型
    # 实际训练神经网络翻译机时数据量肯定是越多越好，不过本文只选取长度小于10的句子
    filtered_pairs = []
    for pair in pairs:
        if len(pair[0]) < MAX_LEN and len(pair[1]) < MAX_LEN:
            filtered_pairs.append(pair)

    # 创建中英文词表，将中文和因为句子转换成词的ID构成的序列
    # 此外须在词表中添加三个特殊词：<pad>用来对短句子进行填充；<bos>表示解码时的起始符号；<eos>表示解码时的终止符号
    # 在实际任务中，一般还会需要指定<unk>符号表示在词表中未出现过的词，并在构造训练集时有意识地添加<unk>符号，使模型能够处理相应情况
    en_vocab = {}
    cn_vocab = {}
    en_vocab['<pad>'], en_vocab['<bos>'], en_vocab['<eos>'] = 0, 1, 2
    cn_vocab['<pad>'], cn_vocab['<bos>'], cn_vocab['<eos>'] = 0, 1, 2
    en_idx, cn_idx = 3, 3
    for en, cn in filtered_pairs:
        for w in en:
            if w not in en_vocab:
                en_vocab[w] = en_idx
                en_idx += 1
        for w in cn:
            if w not in cn_vocab:
                cn_vocab[w] = cn_idx
                cn_idx += 1

    # 使用<pad>符号将短句子填充成长度一致的句子，便于使用批量数据训练模型
    # 同时根据词表，创建一份实际的用于训练的用numpy array组织起来的数据集
    padded_en_sents = []
    padded_cn_sents = []
    # 训练过程中的预测的目标，即每个中文的当前词去预测下一个词是什么词
    padded_cn_label_sents = []
    for en, cn in filtered_pairs:
        padded_en_sent = en + ['<eos>'] + ['<pad>'] * (MAX_LEN - len(en))
        padded_cn_sent = ['<bos>'] + cn + ['<eos>'] + ['<pad>'] * (MAX_LEN - len(cn))
        padded_cn_label_sent = cn + ['<eos>'] + ['<pad>'] * (MAX_LEN - len(cn) + 1)

        # 根据词表，将相应的单词转换成数字ID
        padded_en_sents.append([en_vocab[w] for w in padded_en_sent])
        padded_cn_sents.append([cn_vocab[w] for w in padded_cn_sent])
        padded_cn_label_sents.append([cn_vocab[w] for w in padded_cn_label_sent])

    # 将训练数据用numpy array组织起来
    train_en_sents = np.array(padded_en_sents, dtype='int64')
    train_cn_sents = np.array(padded_cn_sents, dtype='int64')
    train_cn_label_sents = np.array(padded_cn_label_sents, dtype='int64')

    return train_en_sents, train_cn_sents, train_cn_label_sents, en_vocab, cn_vocab


class Encoder(paddle.nn.Layer):
    """Seq2Seq模型Encoder，采用LSTM结构"""
    def __init__(self, en_vocab_size, en_embedding_dim, lstm_hidden_size, lstm_num_layers):
        super(Encoder, self).__init__()
        self.emb = paddle.nn.Embedding(num_embeddings=en_vocab_size, embedding_dim=en_embedding_dim)
        self.lstm = paddle.nn.LSTM(input_size=en_embedding_dim, hidden_size=lstm_hidden_size,
                                   num_layers=lstm_num_layers)

    def forward(self, x):
        x = self.emb(x)
        x, (h, c) = self.lstm(x)
        return x, h, c


class AttentionDecoder(paddle.nn.Layer):
    """Seq2Seq模型解码器，采用带有注意力机制的LSTM结构"""
    def __init__(self, cn_vocab_size, cn_embedding_dim, lstm_hidden_size, v_dim):
        super(AttentionDecoder, self).__init__()
        self.emb = paddle.nn.Embedding(num_embeddings=cn_vocab_size, embedding_dim=cn_embedding_dim)
        # lstm层输入为x'_t和Context Vector拼接而成的向量，Context Vector的维度与lstm_hidden_size一致
        self.lstm = paddle.nn.LSTM(input_size=cn_embedding_dim + lstm_hidden_size,
                                   hidden_size=lstm_hidden_size)

        # 用于计算Attention权重
        self.attention_linear1 = paddle.nn.Linear(lstm_hidden_size * 2, v_dim)
        self.attention_linear2 = paddle.nn.Linear(v_dim, 1)

        # 用于根据lstm状态计算输出
        self.out_linear = paddle.nn.Linear(lstm_hidden_size, cn_vocab_size)

    # forward函数每次往前计算一次。整体的recurrent部分，是在训练循环内完成的。
    def forward(self, x, previous_hidden, previous_cell, encoder_outputs):
        x = self.emb(x)

        # 对previous_hidden进行数据重排
        hidden_transpose = paddle.transpose(previous_hidden, [1, 0, 2])

        # attention输入：Decoder当前状态和Encoder所有状态
        # 总共需计算MAX_LEN + 1个权重（这是因为会在输入句子后面加上一个<eos>符号）
        attention_inputs = paddle.concat((encoder_outputs,
                                          paddle.tile(hidden_transpose, repeat_times=[1, MAX_LEN + 1, 1])), axis=-1)

        attention_hidden = self.attention_linear1(attention_inputs)
        attention_hidden = F.tanh(attention_hidden)
        attention_logits = self.attention_linear2(attention_hidden)
        attention_logits = paddle.squeeze(attention_logits)

        # 计算得到所有MAX_LEN + 1个权重
        attention_weights = F.softmax(attention_logits)
        attention_weights = paddle.expand_as(paddle.unsqueeze(attention_weights, -1),
                                             encoder_outputs)

        # 计算Context Vector
        context_vector = paddle.multiply(encoder_outputs, attention_weights)
        context_vector = paddle.sum(context_vector, 1)
        context_vector = paddle.unsqueeze(context_vector, 1)

        lstm_input = paddle.concat((x, context_vector), axis=-1)

        x, (hidden, cell) = self.lstm(lstm_input, (previous_hidden, previous_cell))

        output = self.out_linear(hidden)
        output = paddle.squeeze(output)
        return output, (hidden, cell)


def train(train_en_sents, train_cn_sents, train_cn_label_sents, epochs, learning_rate, batch_size,
          en_vocab_size, en_embedding_dim, lstm_hidden_size, lstm_num_layers,
          cn_vocab_size, cn_embedding_dim, v_dim):
    encoder = Encoder(en_vocab_size, en_embedding_dim, lstm_hidden_size, lstm_num_layers)
    atten_decoder = AttentionDecoder(cn_vocab_size, cn_embedding_dim, lstm_hidden_size, v_dim)
    opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=encoder.parameters()+atten_decoder.parameters())

    for epoch in range(epochs):
        print("epoch:{}".format(epoch))

        # 将训练数据集打乱
        perm = np.random.permutation(len(train_en_sents))
        train_en_sents_shuffled = train_en_sents[perm]
        train_cn_sents_shuffled = train_cn_sents[perm]
        train_cn_label_sents_shuffled = train_cn_label_sents[perm]

        for iteration in range(train_en_sents_shuffled.shape[0] // batch_size):
            # 获取一个batch的英文句子训练数据
            x_data = train_en_sents_shuffled[(batch_size * iteration):(batch_size * (iteration + 1))]
            # 将数据转换成paddle内置tensor
            sent = paddle.to_tensor(x_data)
            # 经过encoder得到对输入数据的编码
            en_repr, hidden, cell = encoder(sent)

            # 获取一个batch的对应的中文句子数据
            x_cn_data = train_cn_sents_shuffled[(batch_size * iteration):(batch_size * (iteration + 1))]
            x_cn_label_data = train_cn_label_sents_shuffled[(batch_size * iteration):(batch_size * (iteration + 1))]

            # 损失
            loss = paddle.zeros([1])

            # 解码器循环，计算总损失
            for i in range(MAX_LEN + 2):
                # 获得当前输入atten_decoder的输入元素及标签
                cn_word = paddle.to_tensor(x_cn_data[:, i:i + 1])
                cn_word_label = paddle.to_tensor(x_cn_label_data[:, i])

                logits, (hidden, cell) = atten_decoder(cn_word, hidden, cell, en_repr)
                step_loss = F.cross_entropy(logits, cn_word_label)
                loss += step_loss

            # 计算平均损失
            loss = loss / (MAX_LEN + 2)

            if iteration % 200 == 0:
                print("iter {}, loss:{}".format(iteration, loss.numpy()))

            # 后向传播
            loss.backward()
            # 参数更新
            opt.step()
            # 清除梯度
            opt.clear_grad()

    # 训练完成保存模型参数
    paddle.save(encoder.state_dict(), 'models/encoder.pdparams')
    paddle.save(atten_decoder.state_dict(), 'models/atten_decoder.pdparams')


def translate(num_of_exampels_to_evaluate, encoder, atten_decoder, en_vocab, cn_vocab):
    """
    展示机器翻译效果，用训练集中部分数据查看机器的翻译的结果

    :param num_of_exampels_to_evaluate: 指定从训练多少句子
    :param encoder: 训练好的encoder
    :param atten_decoder: 训练好的atten_decoder
    :param en_vocab: 英文词汇表
    :param cn_vocab: 中文词汇表
    :return: None
    """

    # 将模型设置为eval模式
    encoder.eval()
    atten_decoder.eval()

    # 从训练数据中随机选择部分英语句子展示翻译效果
    indices = np.random.choice(len(train_en_sents), num_of_exampels_to_evaluate, replace=False)
    x_data = train_en_sents[indices]
    sent = paddle.to_tensor(x_data)
    en_repr, hidden, cell = encoder(sent)

    # 获取随机选择到的英语句子和对应的中文翻译
    en_vocab_list = list(en_vocab)
    cn_vocab_list = list(cn_vocab)
    en_sents = []
    cn_sents = []
    for i in range(num_of_exampels_to_evaluate):
        this_en_sents = []
        this_cn_sents = []
        for en_vocab_id in train_en_sents[indices[i]]:
            # 0，1，2是三个特殊符号的ID
            if en_vocab_id not in [0, 1, 2]:
                this_en_sents.append(en_vocab_list[en_vocab_id])
        for cn_vocab_id in train_cn_sents[indices[i]]:
            if cn_vocab_id not in [0, 1, 2]:
                this_cn_sents.append(cn_vocab_list[cn_vocab_id])
        en_sents.append(this_en_sents)
        cn_sents.append(this_cn_sents)

    # Decoder解码时输入的第一个符号为<bos>
    word = np.array([[cn_vocab['<bos>']]] * num_of_exampels_to_evaluate)
    word = paddle.to_tensor(word)

    decoded_sent = []
    for i in range(MAX_LEN + 2):
        logits, (hidden, cell) = atten_decoder(word, hidden, cell, en_repr)
        word = paddle.argmax(logits, axis=1)
        decoded_sent.append(word.numpy())
        word = paddle.unsqueeze(word, axis=-1)

    results = np.stack(decoded_sent, axis=1)
    for i in range(num_of_exampels_to_evaluate):
        en_input = " ".join(en_sents[i][:-1]) + en_sents[i][-1]
        ground_truth_translate = "".join(cn_sents[i][:-1]) + cn_sents[i][-1]
        model_translate = ""
        for k in results[i]:
            w = list(cn_vocab)[k]
            if w != '<pad>' and w != '<eos>':
                model_translate += w
        print(en_input)
        print("true: {}".format(ground_truth_translate))
        print("pred: {}".format(model_translate))


if __name__ == '__main__':
    train_en_sents, train_cn_sents, train_cn_label_sents, en_vocab, cn_vocab = create_dataset('datasets/cmn.txt')

    # 设置超参数
    epochs = 20
    batch_size = 64
    learning_rate = 0.001
    en_vocab_size = len(en_vocab)
    cn_vocab_size = len(cn_vocab)
    en_embedding_dim = 128
    cn_embedding_dim = 128
    lstm_hidden_size = 256
    lstm_num_layers = 1
    v_dim = 256

    start_time = time.time()
    train(train_en_sents, train_cn_sents, train_cn_label_sents, epochs, learning_rate, batch_size,
          en_vocab_size, en_embedding_dim, lstm_hidden_size, lstm_num_layers,
          cn_vocab_size, cn_embedding_dim, v_dim)
    finish_time = time.time()
    print('训练用时：{:.2f}分钟'.format((finish_time-start_time)/60.0))

    # 用训练好的模型来预测
    # 首先创建encoder和atten_decoder对象，并加载训练好的参数
    encoder = Encoder(en_vocab_size, en_embedding_dim, lstm_hidden_size, lstm_num_layers)
    atten_decoder = AttentionDecoder(cn_vocab_size, cn_embedding_dim, lstm_hidden_size, v_dim)
    encoder_state_dict = paddle.load('models/encoder.pdparams')
    atten_decoder_state_dict = paddle.load('models/atten_decoder.pdparams')
    encoder.set_state_dict(encoder_state_dict)
    atten_decoder.set_state_dict(atten_decoder_state_dict)

    # 调用translate函数实现机器翻译——英译中
    translate(10, encoder, atten_decoder, en_vocab, cn_vocab)
