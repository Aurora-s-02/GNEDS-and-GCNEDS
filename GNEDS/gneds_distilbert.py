from __future__ import print_function

import sys

sys.path.append("../")

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchmetrics import HingeLoss
from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, RobertaModel, XLNetPreTrainedModel, \
    XLNetModel, DistilBertPreTrainedModel, DistilBertModel
import math
from collections import OrderedDict, namedtuple, defaultdict
import dgl
import dgl.nn.pytorch as dglnn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

relation_map_ontoevent = {'BEFORE': 1, 'AFTER': 2, 'EQUAL': 3, 'CAUSE': 4, 'CAUSEDBY': 5, 'COSUPER': 6, 'SUBSUPER': 7,
                          'SUPERSUB': 8}
relation_map_mavenere = {'BEFORE': 1, 'OVERLAP': 2, 'CONTAINS': 3, 'SIMULTANEOUS': 4, 'BEGINS-ON': 5, 'ENDS-ON': 6,
                         'CAUSE': 7, 'PRECONDITION': 8, 'subevent_relations': 9, "coreference": 10}
dict_num_sent2rel = {103: len(relation_map_ontoevent), 171: len(relation_map_mavenere)}

ENERGY_WEIGHT = 1
SPC_TOKEN_WEIGHT = 0.1
NA_REL_WEIGHT = 0.1
NA_REL_WEIGHT_TEMP = 0.3
NA_REL_WEIGHT_CAUSAL = 0.02
NA_REL_WEIGHT_SUB = 0.01

class gneds_DistilBert(
    DistilBertPreTrainedModel):  # BertPreTrainedModel, RobertaPreTrainedModel, XLNetPreTrainedModel, DistilBertPreTrainedModel
    def __init__(self, config):
        super().__init__(config)
        self.lm = DistilBertModel(config)
        self.num_labels4token = config.num_labels
        self.num_labels4sent = config.num_labels - 2
        self.relation_size = dict_num_sent2rel[config.num_labels] + 1  # +1 for NA
        self.maxpooling = nn.MaxPool1d(128)
        # self.hidden_dropout_prob = config.hidden_dropout_prob
        self.hidden_dropout_prob = 0.2
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.aggr = "task_based"  # task_based, mean, max, max_pooling

        # self.attention = Attention(hidden_size*2,2)
        # some hyperparameters
        # 损失值？ plus？看看~
        # 根据计算不同方向时需要更改，在OutoEvent-Doc数据集时，trigger：1,0.1,0.1  Event：0.1,1,0.1  Doc：0.1,0.1,1
        self.ratio_loss_token_plus = 1  # \mu_1
        self.ratio_loss_token = 1  # \lambda_1
        self.ratio_loss_sent_plus = 1  # \mu_2
        self.ratio_loss_sent = 0.1  # \lambda_2
        self.ratio_loss_doc_plus = 1  # \mu_3
        self.ratio_loss_doc = 0.1  # \lambda_3
        self.attention = Attention(config.hidden_size, 2)
        print("*" * 20, "Speech", "*" * 20)
        print("self.ratio_loss_token_plus", self.ratio_loss_token_plus)
        print("self.ratio_loss_token", self.ratio_loss_token)
        print("self.ratio_loss_sent_plus", self.ratio_loss_sent_plus)
        print("self.ratio_loss_sent", self.ratio_loss_sent)
        print("self.ratio_loss_doc_plus", self.ratio_loss_doc_plus)
        print("self.ratio_loss_doc", self.ratio_loss_doc)
        self.token = Token(self.num_labels4token, config.hidden_size, self.hidden_dropout_prob,
                           self.ratio_loss_token_plus)  # hidden_size(隐藏层维度,隐藏层的神经元个数)
        self.sent = Sentence(self.num_labels4sent, config.hidden_size, self.hidden_dropout_prob,
                             self.ratio_loss_sent_plus)
        self.doc = Document(self.relation_size, config.hidden_size, self.hidden_dropout_prob, self.ratio_loss_doc_plus)

        self.init_weights()

    def get_pos_in_batch(num, list_num, max_mention_size):
        """ num: the reconstructed pos in the real batch (the real batch size is a sum of real mention sizes)
            list_num: the list of real mention size
            max_mention_size: the maximum number of event mentions in one doc
            return: the pos index in the padding normalized batch whose size is [batch_size, max_size]
        """
        batch_size = list_num.size(0)
        if batch_size == 1 or num <= list_num[0].item():
            return 0, num
        sum_num = 0
        for i in range(batch_size - 1):
            sum_num += min(list_num[i].item(), max_mention_size)
            if sum_num < num <= sum_num + min(list_num[i + 1].item(), max_mention_size):
                return i + 1, num - sum_num - 1

    def forward(self, example_id=None, task_name=None, doc_ere_task_type=None, max_mention_size=None,
                pad_token_label_id=None, input_ids=None, input_dependent=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, mention_size=None, labels4token=None,
                labels4sent=None, mat_rel_label=None):
        batch_size = int(input_ids.size(0) / max_mention_size[0].item())
        num_or_max_mention = max_mention_size[0].item()
        max_seq_length = input_ids.size(1)
        if_special = 0
        if batch_size < math.ceil(
                input_ids.size(0) / max_mention_size[0].item()):  # abnormal......may happen in the last batch?
            if_special = 1
            batch_size = 1  # regard the rest samples exist in one batch, note that their labels, mention_size and doc_token_emb should also be reshaped
            num_or_max_mention = input_ids.size(0)
            real_batch_size = mat_rel_label.size(0)
            real_max_mention = mat_rel_label.size(1)
            mention_size_rebuilt = torch.ones([1], dtype=torch.long).to(device)
            labels4token_rebuilt = (
                        torch.ones([1, num_or_max_mention, max_seq_length], dtype=torch.long) * pad_token_label_id[
                    0].item()).to(device)
            labels4sent_rebuilt = (
                        torch.ones([1, num_or_max_mention], dtype=torch.long) * pad_token_label_id[0].item()).to(device)
            mat_rel_label_rebulit = torch.zeros([1, num_or_max_mention, num_or_max_mention], dtype=torch.long).to(
                device)
            count_num_mention = 0
            for i in range(real_batch_size):
                real_num_mention = min(mention_size[i].item(), real_max_mention)
                real_num_mention = min(real_num_mention, num_or_max_mention)
                real_num_mention = min(real_num_mention, num_or_max_mention - i * real_max_mention)
                labels4token_rebuilt[0, count_num_mention: count_num_mention + real_num_mention, :] = labels4token[i,
                                                                                                      :real_num_mention,
                                                                                                      :]
                labels4sent_rebuilt[0, count_num_mention: count_num_mention + real_num_mention] = labels4sent[i,
                                                                                                  :real_num_mention]
                mat_rel_label_rebulit[0, count_num_mention: count_num_mention + real_num_mention,
                count_num_mention: count_num_mention + real_num_mention] = mat_rel_label[i, :real_num_mention,
                                                                           :real_num_mention]
                count_num_mention += real_num_mention
            mention_size_rebuilt[0] = count_num_mention

        outputs = self.lm(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        input_dependent = input_dependent
        doc_token_embed = outputs[0].view(batch_size, num_or_max_mention, max_seq_length,
                                          -1)  # [batch_size, max_size, max_length, hidden_size]
        doc_token_embed_01 = doc_token_embed[0]
        doc_token_embed_01_real, token_dependent = self.attention(doc_token_embed_01, 128, input_dependent)

        if if_special == 1:
            doc_token_embed_rebuilt = doc_token_embed.clone()
            real_batch_size = mat_rel_label.size(0)
            real_max_mention = mat_rel_label.size(1)
            count_num_mention = 0
            for i in range(real_batch_size):
                real_num_mention = min(mention_size[i].item(), real_max_mention)
                real_num_mention = min(real_num_mention, num_or_max_mention)
                real_num_mention = min(real_num_mention, num_or_max_mention - i * real_max_mention)
                doc_token_embed_rebuilt[0, count_num_mention: count_num_mention + real_num_mention, :,
                :] = doc_token_embed[:, real_max_mention * i: real_max_mention * i + real_num_mention, :, :]
                count_num_mention += real_num_mention

            mention_size = mention_size_rebuilt
            labels4token = labels4token_rebuilt
            labels4sent = labels4sent_rebuilt
            mat_rel_label = mat_rel_label_rebulit
            doc_token_embed = doc_token_embed_rebuilt.clone()

        if labels4token is not None:
            loss_token, logits_token, token_labels_real, doc_token_embed_real = self.token(doc_token_embed,
                                                                                           labels4token, mention_size,
                                                                                           attention_mask,
                                                                                           pad_token_label_id,
                                                                                           input_dependent)  # 训练触发词的能量
            outputs = (logits_token, token_labels_real,) + outputs[2:]
            # get sentence embedding
            # # for max_pooling
            # doc_sent_embed = self.maxpooling(doc_token_embed.view(batch_size*num_or_max_mention, max_seq_length, -1).transpose(1, 2)).contiguous().view(batch_size, num_or_max_mention, self.config.hidden_size)
            # doc_sent_embed = F.relu(doc_sent_embed) # [batch_size, max_size, hidden_size]
            # # for task_based

            doc_sent_embed = doc_token_embed_01_real[:, :, 0, :]  # [batch_size, max_size, hidden_size]
            # x_reshaped = doc_token_embed_01_real.view(-1, 128, 768)
            #
            # # Create an instance of the SelfAttention module
            # atten = SelfAttention(k=768, heads=8)
            #
            # # Apply self-attention to the reshaped input
            # attended_x = atten(x_reshaped)
            #
            # # Since we've merged the first two dimensions for attention,
            # # the output will be of shape (50, 128, 768). We should sum or
            # # average over the second dimension (the one with size 128).
            # # Let's use mean pooling over the second dimension.
            # pooled_x = attended_x.mean(dim=1)
            #
            # # Reshape to get the final output
            # output = pooled_x.view(1, 50, 768)
            if self.aggr == "task_based":
                indices_trigger_token = (
                            labels4token < self.num_labels4sent - 2).nonzero()  # not non-trigger or padding
                for trigger_index in indices_trigger_token:
                    if trigger_index[0] < doc_sent_embed.size(0) and trigger_index[1] < doc_sent_embed.size(1) and \
                            trigger_index[2] < max_seq_length:
                        doc_sent_embed[trigger_index[0]][trigger_index[1]] = \
                        doc_token_embed[trigger_index[0]][trigger_index[1]][trigger_index[2]]
            elif self.aggr == "mean" or self.aggr == "max":
                for i in range(batch_size):
                    for j in range(num_or_max_mention):
                        index_valid_token = torch.nonzero(torch.lt(labels4token, pad_token_label_id[0].item())).reshape(
                            -1)
                        tensor_valid_token = doc_token_embed[i, j, index_valid_token, :]
                        if self.aggr == "mean":
                            doc_sent_embed[i, j, :] = tensor_valid_token.mean(0)
                        elif self.aggr == "max":
                            doc_sent_embed[i, j, :] = tensor_valid_token.max(0)[0]
            elif self.aggr == "max_pooling":
                doc_sent_embed = self.maxpooling(
                    doc_token_embed.view(batch_size * num_or_max_mention, max_seq_length, -1).transpose(1,
                                                                                                        2)).contiguous().view(
                    batch_size, num_or_max_mention, self.config.hidden_size)
                doc_sent_embed = F.relu(doc_sent_embed)  # [batch_size, max_size, hidden_size]
            # doc_sent_embed = self.dropout(doc_sent_embed)
            if labels4sent is not None:
                loss_sent, logits_sent, labels_sent_real, proto_embed = self.sent(doc_sent_embed, labels4sent,
                                                                                  mention_size)  # 句子能量训练
                outputs = (logits_sent, labels_sent_real,) + outputs
                if mat_rel_label is not None:
                    if doc_ere_task_type != "doc_joint":
                        loss_doc, logits_sentpair, labels_doc = self.doc(doc_sent_embed, doc_token_embed_01_real,
                                                                         mat_rel_label, mention_size, task_name,
                                                                         doc_ere_task_type)
                        outputs = (logits_sentpair, labels_doc,) + outputs
                    else:
                        if task_name == "maven-ere":
                            loss_doc, logits_sentpair_temp, labels_sentpair_temporal, logits_sentpair_causal, labels_sentpair_causal, logits_sentpair_sub, labels_sentpair_sub, logits_sentpair_corref, labels_sentpair_corref = self.doc(
                                doc_sent_embed, doc_token_embed_01_real, mat_rel_label, mention_size, task_name,
                                doc_ere_task_type)
                            outputs = (logits_sentpair_temp, labels_sentpair_temporal, logits_sentpair_causal,
                                       labels_sentpair_causal, logits_sentpair_sub, labels_sentpair_sub,
                                       logits_sentpair_corref, labels_sentpair_corref,) + outputs
                        else:
                            loss_doc, logits_sentpair_temp, labels_sentpair_temporal, logits_sentpair_causal, labels_sentpair_causal, logits_sentpair_sub, labels_sentpair_sub = self.doc(
                                doc_sent_embed, doc_token_embed_01_real, mat_rel_label, mention_size, task_name,
                                doc_ere_task_type)
                            outputs = (logits_sentpair_temp, labels_sentpair_temporal, logits_sentpair_causal,
                                       labels_sentpair_causal, logits_sentpair_sub, labels_sentpair_sub,) + outputs
                    loss_all = self.ratio_loss_token * loss_token + self.ratio_loss_sent * loss_sent + self.ratio_loss_doc * loss_doc

        torch.autograd.set_detect_anomaly(True)

        outputs = (loss_all,) + outputs

        return outputs


# 构建token的神经网络
class Token(nn.Module):
    def __init__(self, tokentype_size, hidden_size, hidden_dropout_prob, ratio_loss_token_plus):
        super(Token, self).__init__()
        self.tokentype_size = tokentype_size  # 103
        self.ratio_loss_token_plus = ratio_loss_token_plus  # 1
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.attention = Attention(hidden_size, 2)
        self.token_classifier = nn.Linear(hidden_size, self.tokentype_size)  # 输入的神经元个数为hidden_size 输出的个数为103 只有一层隐藏层？
        # ？
        self.mat_local4token = nn.Embedding(self.tokentype_size, hidden_size).to(
            device)  # nn.Embedding((num_embeddings,embedding_dim) 字典大小为103，维度为768
        self.mat_label4token = nn.Embedding(self.tokentype_size, self.tokentype_size).to(device)  # 103,103

    def get_para_vec_mat(self, para_type):  # 获取参数向量图
        mat_local4token = self.mat_local4token(torch.tensor(range(0, self.tokentype_size)).to(device))
        mat_label4token = self.mat_label4token(torch.tensor(range(0, self.tokentype_size)).to(device))
        if para_type == "mat_local":
            return mat_local4token
        else:
            return mat_label4token

    def calculate_prob(self, token_embed):
        # token_embed = self.dropout(token_embed)
        # logits_token = F.softmax(self.token_classifier(token_embed)) # 分类器
        x = self.token_classifier(token_embed)
        logits_token = F.softmax(self.token_classifier(token_embed))
        return logits_token  # 18*128*103

    def token_energy_function(self, token_embed, token_y):  # 词的能量计算 token_y  分类的结果
        token_local_energy_temp = torch.matmul(self.get_para_vec_mat("mat_local"),
                                               token_embed.transpose(1, 2))  # 18*103*128  18*  token_embed  18*128*768
        token_local_energy = torch.sum(torch.mul(token_y, token_local_energy_temp.transpose(1, 2)))
        batch_size = token_y.size(0)
        seq_length = token_y.size(1)
        for i in range(seq_length - 1):
            token_label_energy = torch.sum(
                torch.matmul(torch.matmul(token_y[:, i, :], self.get_para_vec_mat("mat_label")),
                             token_y[:, i + 1, :].transpose(0, 1)))
        token_energy = token_local_energy + token_label_energy
        return token_energy

    def label2vec(self, label, label_size):
        batch_size = label.size(0)
        seq_len = label.size(1)
        label_vec = torch.zeros([batch_size, seq_len, label_size]).to(device)
        for i in range(batch_size):
            for j in range(seq_len):
                label_vec[i][j][label[i][j]] = 1
        return label_vec

    def get_the_real_token_task(self, token_embed, token_labels, mention_size, attention_mask,
                                input_dependent):  # 获得真实值
        batch_size = token_embed.size(0)
        max_mention_size = token_embed.size(1)
        max_seq_length = token_embed.size(2)
        hidden_size = token_embed.size(3)
        attention_mask = attention_mask.view(batch_size, max_mention_size, max_seq_length)
        input_dependent = input_dependent.view(batch_size, max_mention_size, max_seq_length)
        num_mention = 0
        norm_mention_size = [max_mention_size] * batch_size
        for i in range(batch_size):
            norm_mention_size[i] = min(mention_size[i].item(), max_mention_size)
            num_mention += norm_mention_size[i]
        token_embed_real = torch.zeros([num_mention, max_seq_length, hidden_size], dtype=torch.float).to(device)
        token_labels_real = torch.zeros([num_mention, max_seq_length], dtype=torch.long).to(device)
        attention_mask_real = torch.zeros([num_mention, max_seq_length], dtype=torch.float).to(device)
        token_dependent = torch.zeros([num_mention, max_seq_length], dtype=torch.float).to(device)
        count_mention = 0
        for i in range(batch_size):
            token_embed_real[count_mention:count_mention + norm_mention_size[i], :, :] = token_embed[i,
                                                                                         :norm_mention_size[i], :, :]
            token_labels_real[count_mention:count_mention + norm_mention_size[i], :] = token_labels[i,
                                                                                       :norm_mention_size[i], :]
            attention_mask_real[count_mention:count_mention + norm_mention_size[i], :] = attention_mask[i,
                                                                                         :norm_mention_size[i], :]
            token_dependent[count_mention:count_mention + norm_mention_size[i], :] = input_dependent[i,
                                                                                     :norm_mention_size[i], :]
            count_mention += norm_mention_size[i]
        return token_embed_real, token_labels_real, attention_mask_real, token_dependent

    def forward(self, token_embed, token_labels, mention_size, attention_mask, pad_token_label_id, input_dependent):
        token_embed_real, token_labels_real, attention_mask_real, token_dependent = self.get_the_real_token_task(
            token_embed, token_labels, mention_size, attention_mask, input_dependent)
        logits_token = self.calculate_prob(token_embed_real)

        if token_labels_real is not None:
            loss_hinge = HingeLoss(ignore_index=pad_token_label_id[
                0].item())  # [self.tokentype_size-2, self.tokentype_size-1], self.tokentype_size-1==pad_token_label_id[0].item()
            loss_token_hinge = loss_hinge(logits_token.view(-1, self.tokentype_size), token_labels_real.view(-1))
            label_vec = self.label2vec(token_labels_real, self.tokentype_size)
            _, pred_token = torch.max(logits_token, dim=2)
            pred_vec = self.label2vec(pred_token, self.tokentype_size)
            loss_token_energy = torch.max(torch.tensor([0,
                                                        loss_token_hinge + self.token_energy_function(token_embed_real,
                                                                                                      label_vec) - self.token_energy_function(
                                                            token_embed_real, pred_vec)],
                                                       dtype=torch.float))  # 触发词能量损失函数

            # # ignore redundant padding tokens
            logits_token = logits_token.view(-1, self.tokentype_size)
            token_labels_real = token_labels_real.view(-1)
            valid_token_indice = torch.nonzero(torch.ne(token_labels_real, pad_token_label_id[0].item()))[:, 0]
            logits_token_valid = torch.zeros([valid_token_indice.size(0) + 2, self.tokentype_size],
                                             dtype=torch.float).to(device)
            token_labels_real_valid = torch.zeros([valid_token_indice.size(0) + 2], dtype=torch.long).to(device)
            logits_token_valid[[0, -1], :] = logits_token[[0, -1], :]
            token_labels_real_valid[[0, -1]] = token_labels_real[[0, -1]]
            if valid_token_indice.size(0) > 1:
                logits_token_valid[1:-1, :] = logits_token[valid_token_indice, :]
                token_labels_real_valid[1:-1] = token_labels_real[valid_token_indice]
            else:
                logits_token_valid = logits_token
                token_labels_real_valid = token_labels_real
            loss_fct = CrossEntropyLoss(ignore_index=pad_token_label_id[0].item())
            loss_token_plus = loss_fct(logits_token.view(-1, self.tokentype_size), token_labels_real.view(-1))
            loss_token = ENERGY_WEIGHT * loss_token_energy + self.ratio_loss_token_plus * loss_token_plus

        return loss_token, logits_token_valid, token_labels_real_valid, token_embed_real


# 构建sentence的神经网络
class Sentence(nn.Module):
    def __init__(self, proto_size, hidden_size, hidden_dropout_prob, ratio_loss_sent_plus):
        super(Sentence, self).__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.maxpooling = nn.MaxPool1d(128)
        self.prototypes = nn.Embedding(proto_size, hidden_size).to(device)  # 101,768
        self.mat_local4sent = nn.Embedding(proto_size, hidden_size).to(device)  # 101,768
        self.vec_label4sent = nn.Embedding(proto_size, 1).to(device)  # 101,1
        self.mat_label4sent = nn.Embedding(proto_size, proto_size).to(device)  # 101,101
        self.classifier = nn.Linear(hidden_size, proto_size)  # 768,101
        self.proto_size = proto_size  # 101
        self.hidden_size = hidden_size  # 768
        self.ratio_loss_sent_plus = ratio_loss_sent_plus  # 1

    def get_proto_embedding(self):
        proto_embedding = self.prototypes(torch.tensor(range(0, self.proto_size)).to(device))
        return proto_embedding  # [proto_size, hidden_size]

    def get_para_vec_mat(self, para_type):
        mat_local4sent = self.mat_local4sent(torch.tensor(range(0, self.proto_size)).to(device))
        vec_label4sent = self.vec_label4sent(torch.tensor(range(0, self.proto_size)).to(device))
        mat_label4sent = self.mat_label4sent(torch.tensor(range(0, self.proto_size)).to(device))
        if para_type == "mat_local":
            return mat_local4sent
        elif para_type == "vec_label":
            return vec_label4sent
        else:
            return mat_label4sent

    def __dist__(self, x, y, dim):
        dist = torch.pow(x - y, 2).sum(dim)
        # dist = torch.where(torch.isnan(dist), torch.full_like(dist, 1e-8), dist)
        return dist

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)

    def measurement(self, r, p, x):
        return - torch.max([0, self.__dist__(p, x, 2) - r])

    def batch_measurement(self, r, P, X):
        batch_size = X.size(0)
        proto_size = P.size(0)
        return - torch.maximum(torch.zeros([batch_size, proto_size]).to(device),
                               self.__dist__(P.unsqueeze(0), X.unsqueeze(1), 2) - r)

    def calculate_prob(self, r, P, X):
        return F.softmax(self.batch_measurement(r, P, X))
        # return F.relu(self.batch_measurement(r, P, X))
        # return self.batch_measurement(r, P, X)

    def label2vec(self, label, label_size):
        batch_size = label.size(0)
        label_vec = torch.zeros([batch_size, label_size]).to(device)
        for i in range(batch_size):
            label_vec[i][label[i]] = 1
        return label_vec

    def sent_energy_function(self, sent_emb, sent_y):
        sent_local_energy_temp = torch.matmul(self.get_para_vec_mat("mat_local"), sent_emb.transpose(0, 1))
        sent_local_energy = torch.sum(torch.mul(sent_y, sent_local_energy_temp.transpose(0, 1)))
        sent_label_energy = torch.sum(torch.matmul(self.get_para_vec_mat("vec_label").transpose(0, 1), torch.sigmoid(
            torch.matmul(self.get_para_vec_mat("mat_label"), sent_y.transpose(0, 1)))))
        sent_energy = sent_local_energy + sent_label_energy
        return sent_energy

    def get_the_real_sent_task(self, sent_embed, sent_labels, mention_size):
        batch_size = sent_embed.size(0)
        max_mention_size = sent_embed.size(1)
        hidden_size = sent_embed.size(2)
        num_mention = 0
        norm_mention_size = [max_mention_size] * batch_size
        for i in range(batch_size):
            norm_mention_size[i] = min(mention_size[i].item(), max_mention_size)
            num_mention += norm_mention_size[i]
        sent_embed_real = torch.zeros([num_mention, hidden_size], dtype=torch.float).to(device)
        sent_labels_real = torch.zeros([num_mention], dtype=torch.long).to(device)
        count_mention = 0
        for i in range(batch_size):
            sent_embed_real[count_mention:count_mention + norm_mention_size[i], :] = sent_embed[i,
                                                                                     :norm_mention_size[i], :]
            sent_labels_real[count_mention:count_mention + norm_mention_size[i]] = sent_labels[i, :norm_mention_size[i]]
            count_mention += norm_mention_size[i]
        return sent_embed_real, sent_labels_real

    def forward(self, sent_embed, sent_labels, mention_size):
        sent_embed_real, sent_labels_real = self.get_the_real_sent_task(sent_embed, sent_labels, mention_size)
        proto_embed = self.get_proto_embedding()
        logits_sent = self.calculate_prob(1, proto_embed, sent_embed_real)

        if sent_labels_real is not None:
            loss_hinge = HingeLoss()  # ignore_index=0
            loss_sent_hinge = loss_hinge(logits_sent.view(-1, self.proto_size), sent_labels_real.view(-1))
            label_vec = self.label2vec(sent_labels_real, self.proto_size)
            loss_sent_energy = torch.max(torch.tensor([0, loss_sent_hinge + self.sent_energy_function(sent_embed_real,
                                                                                                      label_vec) - self.sent_energy_function(
                sent_embed_real, logits_sent)], dtype=torch.float))
            loss_fct = CrossEntropyLoss()
            loss_sent_plus = loss_fct(logits_sent.view(-1, self.proto_size), sent_labels_real.view(-1))
            loss_sent = ENERGY_WEIGHT * loss_sent_energy + self.ratio_loss_sent_plus * loss_sent_plus

        return loss_sent, logits_sent, sent_labels_real, proto_embed

class Document(nn.Module):
    def __init__(self, relation_size, hidden_size, hidden_dropout_prob, ratio_loss_doc_plus):
        super(Document, self).__init__()
        self.relation_size = relation_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.ratio_loss_doc_plus = ratio_loss_doc_plus
        # self.ere_classifier = nn.Linear(hidden_size*4, relation_size)
        self.dim_expand = 3
        self.ere_classifier = nn.Linear(hidden_size * self.dim_expand, relation_size)

        self.sent_embedding = nn.Parameter(torch.randn(hidden_size))
        self.ere_classifier_joint = nn.Linear(hidden_size * self.dim_expand, relation_size)
        self.ere_classifier_temp_onto = nn.Linear(hidden_size * self.dim_expand, 1 + 3)
        self.ere_classifier_causal_onto = nn.Linear(hidden_size * self.dim_expand, 1 + 2)
        self.ere_classifier_sub_onto = nn.Linear(hidden_size * self.dim_expand, 1 + 3)
        self.ere_classifier_temp_maven = nn.Linear(hidden_size * self.dim_expand, 1 + 6)
        self.ere_classifier_causal_maven = nn.Linear(hidden_size * self.dim_expand, 1 + 2)
        self.ere_classifier_sub_maven = nn.Linear(hidden_size * self.dim_expand, 1 + 1)
        self.ere_classifier_corref_maven = nn.Linear(hidden_size * self.dim_expand, 1 + 1)

        self.mat_local4doc = nn.Embedding(relation_size, hidden_size * self.dim_expand).to(device)
        self.vec_label4doc = nn.Embedding(relation_size, 1).to(device)
        self.mat_label4doc = nn.Embedding(relation_size, relation_size).to(device)

    def get_para_vec_mat(self, para_type, list_ids):
        # list_ids = list(range(0, self.relation_size))
        mat_local4doc = self.mat_local4doc(torch.tensor(list_ids).to(device))
        vec_label4doc = self.vec_label4doc(torch.tensor(list_ids).to(device))
        mat_label4doc = self.mat_label4doc(torch.tensor(list_ids).to(device))[:, list_ids]
        if para_type == "mat_local":
            return mat_local4doc
        elif para_type == "vec_label":
            return vec_label4doc
        else:
            return mat_label4doc

    def get_embedding_interaction(self, t1, t2):
        if self.dim_expand == 2:
            return torch.cat([t1, t2], dim=0)
        elif self.dim_expand == 3:
            return torch.cat([t1, t2, torch.mul(t1, t2)], dim=0)  # we choose this one
        elif self.dim_expand == 4:
            return torch.cat([t1, t2, torch.mul(t1, t2), t1 - t2], dim=0)

    def label2vec(self, label, label_size):
        batch_size = label.size(0)
        label_vec = torch.zeros([batch_size, label_size]).to(device)
        for i in range(batch_size):
            label_vec[i][label[i]] = 1
        return label_vec

    def doc_energy_function(self, X, Y, list_ids):
        doc_local_energy_temp = torch.matmul(self.get_para_vec_mat("mat_local", list_ids), X.transpose(0, 1))
        doc_local_energy = torch.sum(torch.mul(Y, doc_local_energy_temp.transpose(0, 1)))
        doc_label_energy = torch.sum(torch.matmul(self.get_para_vec_mat("vec_label", list_ids).transpose(0, 1),
                                                  torch.sigmoid(
                                                      torch.matmul(self.get_para_vec_mat("mat_label", list_ids),
                                                                   Y.transpose(0, 1)))))
        doc_energy = doc_local_energy + doc_label_energy
        return doc_energy

    def get_event_re_task(self, sent_embed, mat_rel_label, mention_size, task_name, doc_ere_task_type):
        batch_size = sent_embed.size(0)
        max_mention_size = sent_embed.size(1)
        hidden_size = sent_embed.size(2)
        num_rel = self.relation_size
        num_mention = 0
        num_mention_pair = 0
        norm_mention_size = [max_mention_size] * batch_size
        for i in range(batch_size):
            norm_mention_size[i] = min(mention_size[i].item(), max_mention_size)
            num_mention += norm_mention_size[i]
            if norm_mention_size[i] != 1:
                num_mention_pair += norm_mention_size[i] * (norm_mention_size[i] - 1)
            else:
                num_mention_pair += 1

        inputs_sentpair = torch.zeros([num_mention_pair, hidden_size * self.dim_expand], dtype=torch.float).to(device)
        labels_sentpair = torch.zeros([num_mention_pair], dtype=torch.long).to(device)

        count_example_pair = 0
        for k in range(batch_size):
            num_mention_one_doc = norm_mention_size[k]
            if num_mention_one_doc != 1:
                for i in range(num_mention_one_doc):
                    for j in range(num_mention_one_doc):
                        if i != j:
                            inputs_sentpair[count_example_pair] = self.get_embedding_interaction(sent_embed[k][i],
                                                                                                 sent_embed[k][j])
                            labels_sentpair[count_example_pair] = mat_rel_label[k][i][j].item()
                            count_example_pair += 1
            else:
                inputs_sentpair[count_example_pair] = self.get_embedding_interaction(sent_embed[k][0], sent_embed[k][0])
                labels_sentpair[count_example_pair] = mat_rel_label[k][0][0].item()
                count_example_pair += 1

        if doc_ere_task_type == "doc_all":
            return inputs_sentpair, labels_sentpair
        else:
            if task_name == "ontoevent-doc":
                labels_sentpair_temporal, labels_sentpair_causal, labels_sentpair_sub = self.labels_sentpair_rebuilt(
                    labels_sentpair, task_name)
                return inputs_sentpair, labels_sentpair_temporal, labels_sentpair_causal, labels_sentpair_sub
            elif task_name == "maven-ere":
                labels_sentpair_temporal, labels_sentpair_causal, labels_sentpair_sub, labels_sentpair_corref = self.labels_sentpair_rebuilt(
                    labels_sentpair, task_name)
                return inputs_sentpair, labels_sentpair_temporal, labels_sentpair_causal, labels_sentpair_sub, labels_sentpair_corref

    def labels_sentpair_rebuilt(self, labels_sentpair, task_name):
        # rebuild the labels_sentpair for different ere task on different dataset
        labels_sentpair_temporal = labels_sentpair.clone()
        labels_sentpair_causal = labels_sentpair.clone()
        labels_sentpair_sub = labels_sentpair.clone()
        labels_sentpair_corref = labels_sentpair.clone()
        label_size = labels_sentpair.size(0)

        if task_name == "maven-ere":
            for i in range(label_size):
                label = labels_sentpair[i].item()
                if label not in list(range(1, 7)):
                    labels_sentpair_temporal[i] = 0

                if label not in list(range(7, 9)):
                    labels_sentpair_causal[i] = 0
                else:
                    labels_sentpair_causal[i] = labels_sentpair_causal[i] - 6

                if label != 9:
                    labels_sentpair_sub[i] = 0
                else:
                    labels_sentpair_sub[i] = labels_sentpair_sub[i] - 8

                if label != 10:
                    labels_sentpair_corref[i] = 0
                else:
                    labels_sentpair_corref[i] = labels_sentpair_corref[i] - 9

            return labels_sentpair_temporal, labels_sentpair_causal, labels_sentpair_sub, labels_sentpair_corref

        elif task_name == "ontoevent-doc":
            for i in range(label_size):
                label = labels_sentpair[i].item()
                if label not in list(range(1, 4)):
                    labels_sentpair_temporal[i] = 0

                if label not in list(range(4, 6)):
                    labels_sentpair_causal[i] = 0
                else:
                    labels_sentpair_causal[i] = labels_sentpair_causal[i] - 3

                if label not in list(range(6, 9)):
                    labels_sentpair_sub[i] = 0
                else:
                    labels_sentpair_sub[i] = labels_sentpair_sub[i] - 5

            return labels_sentpair_temporal, labels_sentpair_causal, labels_sentpair_sub

    def calculate_ere_loss(self, logits_ere, labels_ere, sentpair_emb, relation_size, list_ids, task_name,
                           doc_ere_task_type):
        if labels_ere is not None:
            loss_hinge = HingeLoss()  # ignore_index=0, num_classes=relation_size
            loss_doc_hinge = loss_hinge(logits_ere.view(-1, relation_size), labels_ere.view(-1))
            label_vec = self.label2vec(labels_ere, relation_size)
            loss_doc_energy = torch.max(torch.tensor([0,
                                                      loss_doc_hinge + self.doc_energy_function(sentpair_emb, label_vec,
                                                                                                list_ids) - self.doc_energy_function(
                                                          sentpair_emb, logits_ere, list_ids)], dtype=torch.float))
            if doc_ere_task_type == "doc_all" or (task_name == "ontoevent-doc" and doc_ere_task_type != "doc_causal"):
                weight_tensor = torch.ones([relation_size]).to(device)
                weight_tensor[
                    0] = NA_REL_WEIGHT  # as there are too many NA relations, we should decrease their weight in loss and focus more on valid labels' training
                weight_tensor = weight_tensor / torch.sum(weight_tensor)
                loss_fct = CrossEntropyLoss(weight=weight_tensor)  # , ignore_index=0
                # loss_fct = CrossEntropyLoss()
            elif task_name == "ontoevent-doc" and doc_ere_task_type == "doc_causal":
                weight_tensor = torch.ones([relation_size]).to(device)
                weight_tensor[0] = NA_REL_WEIGHT / 2
                weight_tensor = weight_tensor / torch.sum(weight_tensor)
                loss_fct = CrossEntropyLoss(weight=weight_tensor)  # , ignore_index=0
                # loss_fct = CrossEntropyLoss()
            else:
                if task_name == "maven-ere":
                    weight_tensor = torch.ones([relation_size]).to(device)
                    if doc_ere_task_type == "doc_sub":
                        weight_tensor[0] = NA_REL_WEIGHT_SUB
                    elif doc_ere_task_type == "doc_temporal":
                        weight_tensor[0] = NA_REL_WEIGHT_TEMP
                    elif doc_ere_task_type == "doc_causal":
                        weight_tensor[0] = NA_REL_WEIGHT_CAUSAL
                    weight_tensor = weight_tensor / torch.sum(weight_tensor)
                    loss_fct = CrossEntropyLoss(weight=weight_tensor)  # , ignore_index=0
            loss_doc_plus = loss_fct(logits_ere.view(-1, relation_size) + 1e-10,
                                     labels_ere.view(-1))  # +1e-10 to avoid nan in loss

            loss_doc = ENERGY_WEIGHT * loss_doc_energy + self.ratio_loss_doc_plus * loss_doc_plus

            return loss_doc

    def forward(self, sent_embed, doc_token_embed, mat_rel_label, mention_size, task_name, doc_ere_task_type):

        if doc_ere_task_type == "doc_all":
            sentpair_emb, labels_sentpair = self.get_event_re_task(sent_embed, mat_rel_label, mention_size,
                                                                   task_name, doc_ere_task_type)
            # logits_sentpair = self.ere_classifier(sentpair_emb) # F.softmax()
            # logits_sentpair_all = F.softmax(self.ere_classifier(sentpair_emb))
            logits_sentpair_all = F.relu(self.ere_classifier(sentpair_emb))
            label_ids = list(range(0, self.relation_size))
            loss_doc_all = self.calculate_ere_loss(logits_sentpair_all, labels_sentpair, sentpair_emb,
                                                   self.relation_size, label_ids, task_name, doc_ere_task_type)
            return loss_doc_all, logits_sentpair_all, labels_sentpair
        if task_name == "maven-ere":
            ratio_temp = 1
            ratio_causal = 2
            ratio_sub = 2
            ratio_corref = 0
            size_temp = 1 + 6  # +1 for NA
            size_causal = 1 + 2  # +1 for NA
            size_sub = 1 + 1  # +1 for NA
            size_corref = 1 + 1  # +1 for NA
            label_temp_ids = list(range(0, size_temp))
            label_causal_ids = [0, 7, 8]
            label_sub_ids = [0, 9]
            label_corref_ids = [0, 10]
            inputs_sentpair, labels_sentpair_temporal, labels_sentpair_causal, labels_sentpair_sub, labels_sentpair_corref = self.get_event_re_task(
                sent_embed, mat_rel_label, mention_size, task_name, doc_ere_task_type)
            if doc_ere_task_type == "doc_temporal":
                # logits_sentpair_temp = F.softmax(self.ere_classifier_temp_maven(inputs_sentpair))
                logits_sentpair_temp = F.relu(self.ere_classifier_temp_maven(inputs_sentpair))
                loss_doc_temp = self.calculate_ere_loss(logits_sentpair_temp, labels_sentpair_temporal, inputs_sentpair,
                                                        size_temp, label_temp_ids, task_name, doc_ere_task_type)
                return loss_doc_temp, logits_sentpair_temp, labels_sentpair_temporal
            elif doc_ere_task_type == "doc_causal":
                # logits_sentpair_causal = F.softmax(self.ere_classifier_causal_maven(inputs_sentpair))
                logits_sentpair_causal = F.relu(self.ere_classifier_causal_maven(inputs_sentpair))
                loss_doc_causal = self.calculate_ere_loss(logits_sentpair_causal, labels_sentpair_causal,
                                                          inputs_sentpair, size_causal, label_causal_ids, task_name,
                                                          doc_ere_task_type)
                return loss_doc_causal, logits_sentpair_causal, labels_sentpair_causal
            elif doc_ere_task_type == "doc_sub":
                # logits_sentpair_sub = F.softmax(self.ere_classifier_sub_maven(inputs_sentpair))
                logits_sentpair_sub = F.relu(self.ere_classifier_sub_maven(inputs_sentpair))
                loss_doc_sub = self.calculate_ere_loss(logits_sentpair_sub, labels_sentpair_sub, inputs_sentpair,
                                                       size_sub, label_sub_ids, task_name, doc_ere_task_type)
                return loss_doc_sub, logits_sentpair_sub, labels_sentpair_sub
            elif doc_ere_task_type == "doc_corref":
                # logits_sentpair_corref = F.softmax(self.ere_classifier_corref_maven(inputs_sentpair))
                logits_sentpair_corref = F.relu(self.ere_classifier_corref_maven(inputs_sentpair))
                loss_doc_corref = self.calculate_ere_loss(logits_sentpair_corref, labels_sentpair_corref,
                                                          inputs_sentpair, size_corref, label_corref_ids, task_name,
                                                          doc_ere_task_type)
                return loss_doc_corref, logits_sentpair_corref, labels_sentpair_corref
            elif doc_ere_task_type == "doc_joint":
                # logits_sentpair_temp = F.softmax(self.ere_classifier_temp_maven(inputs_sentpair))
                # logits_sentpair_causal = F.softmax(self.ere_classifier_causal_maven(inputs_sentpair))
                # logits_sentpair_sub = F.softmax(self.ere_classifier_sub_maven(inputs_sentpair))
                # logits_sentpair_corref = F.softmax(self.ere_classifier_corref_maven(inputs_sentpair))
                logits_sentpair_temp = F.relu(self.ere_classifier_temp_maven(inputs_sentpair))
                logits_sentpair_causal = F.relu(self.ere_classifier_causal_maven(inputs_sentpair))
                logits_sentpair_sub = F.relu(self.ere_classifier_sub_maven(inputs_sentpair))
                logits_sentpair_corref = F.relu(self.ere_classifier_corref_maven(inputs_sentpair))
                loss_doc_temp = self.calculate_ere_loss(logits_sentpair_temp, labels_sentpair_temporal, inputs_sentpair,
                                                        size_temp, label_temp_ids, task_name, "doc_temporal")
                loss_doc_causal = self.calculate_ere_loss(logits_sentpair_causal, labels_sentpair_causal,
                                                          inputs_sentpair, size_causal, label_causal_ids, task_name,
                                                          "doc_causal")
                loss_doc_sub = self.calculate_ere_loss(logits_sentpair_sub, labels_sentpair_sub, inputs_sentpair,
                                                       size_sub, label_sub_ids, task_name, "doc_sub")
                loss_doc_corref = self.calculate_ere_loss(logits_sentpair_corref, labels_sentpair_corref,
                                                          inputs_sentpair, size_corref, label_corref_ids, task_name,
                                                          "doc_corref")
                loss_doc_joint = ratio_temp * loss_doc_temp + ratio_causal * loss_doc_causal + ratio_sub * loss_doc_sub + ratio_corref * loss_doc_corref
                return loss_doc_joint, logits_sentpair_temp, labels_sentpair_temporal, logits_sentpair_causal, labels_sentpair_causal, logits_sentpair_sub, labels_sentpair_sub, logits_sentpair_corref, labels_sentpair_corref
        elif task_name == "ontoevent-doc":
            ratio_temp = 3
            ratio_causal = 1
            ratio_sub = 0
            size_temp = 1 + 3  # +1 for NA
            size_causal = 1 + 2  # +1 for NA
            size_sub = 1 + 3  # +1 for NA
            label_temp_ids = list(range(0, size_temp))
            label_causal_ids = [0, 4, 5]
            label_sub_ids = [0, 6, 7, 8]
            inputs_sentpair, labels_sentpair_temporal, labels_sentpair_causal, labels_sentpair_sub = self.get_event_re_task(
                sent_embed, mat_rel_label, mention_size, task_name, doc_ere_task_type)
            if doc_ere_task_type == "doc_temporal":
                # logits_sentpair_temp = F.softmax(self.ere_classifier_temp_onto(inputs_sentpair))
                logits_sentpair_temp = F.relu(self.ere_classifier_temp_onto(inputs_sentpair))
                loss_doc_temp = self.calculate_ere_loss(logits_sentpair_temp, labels_sentpair_temporal, inputs_sentpair,
                                                        size_temp, label_temp_ids, task_name, doc_ere_task_type)
                return loss_doc_temp, logits_sentpair_temp, labels_sentpair_temporal
            elif doc_ere_task_type == "doc_causal":
                # logits_sentpair_causal = F.softmax(self.ere_classifier_causal_onto(inputs_sentpair))
                logits_sentpair_causal = F.relu(self.ere_classifier_causal_onto(inputs_sentpair))
                loss_doc_causal = self.calculate_ere_loss(logits_sentpair_causal, labels_sentpair_causal,
                                                          inputs_sentpair, size_causal, label_causal_ids, task_name,
                                                          doc_ere_task_type)
                return loss_doc_causal, logits_sentpair_causal, labels_sentpair_causal
            elif doc_ere_task_type == "doc_sub":
                # logits_sentpair_sub = F.softmax(self.ere_classifier_sub_onto(inputs_sentpair))
                logits_sentpair_sub = F.relu(self.ere_classifier_sub_onto(inputs_sentpair))
                loss_doc_sub = self.calculate_ere_loss(logits_sentpair_sub, labels_sentpair_sub, inputs_sentpair,
                                                       size_sub, label_sub_ids, task_name, doc_ere_task_type)
                return loss_doc_sub, logits_sentpair_sub, labels_sentpair_sub
            elif doc_ere_task_type == "doc_joint":
                # logits_sentpair_temp = F.softmax(self.ere_classifier_temp_onto(inputs_sentpair))
                # logits_sentpair_causal = F.softmax(self.ere_classifier_causal_onto(inputs_sentpair))

                # logits_sentpair_sub = F.softmax(self.ere_classifier_sub_onto(inputs_sentpair))
                logits_sentpair_temp = F.relu(self.ere_classifier_temp_onto(inputs_sentpair))
                logits_sentpair_causal = F.relu(self.ere_classifier_causal_onto(inputs_sentpair))
                logits_sentpair_sub = F.relu(self.ere_classifier_sub_onto(inputs_sentpair))
                loss_doc_temp = self.calculate_ere_loss(logits_sentpair_temp, labels_sentpair_temporal, inputs_sentpair,
                                                        size_temp, label_temp_ids, task_name, "doc_temporal")
                loss_doc_causal = self.calculate_ere_loss(logits_sentpair_causal, labels_sentpair_causal,
                                                          inputs_sentpair, size_causal, label_causal_ids, task_name,
                                                          "doc_causal")
                loss_doc_sub = self.calculate_ere_loss(logits_sentpair_sub, labels_sentpair_sub, inputs_sentpair,
                                                       size_sub, label_sub_ids, task_name, "doc_sub")
                loss_doc_joint = ratio_temp * loss_doc_temp + ratio_causal * loss_doc_causal + ratio_sub * loss_doc_sub
                return loss_doc_joint, logits_sentpair_temp, labels_sentpair_temporal, logits_sentpair_causal, labels_sentpair_causal, logits_sentpair_sub, labels_sentpair_sub


class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x, max_seq_length, token_dependent=None):
        # batch_size = x.size(0)
        # for i in range(batch_size):
        if token_dependent is None:
            scores = self.v(torch.tanh(self.W(x)))
            token_dependent = torch.softmax(scores, dim=1)
        else:
            for i, inner_list in enumerate(token_dependent):
                token_dependent[i] = inner_list * 100

            max_length = max_seq_length
            token_dependent_padded = []
            for inner_tensor in token_dependent:
                reshaped_tensor = inner_tensor.view(1, -1, 1)
                token_dependent_padded.append(reshaped_tensor)

            token_dependent = torch.stack(token_dependent_padded, dim=0).to("cuda:0")
        head_outputs = []
        for _ in range(self.num_heads):
            scores = self.v(torch.tanh(self.W(x)))
            # token_dependent = torch.relu(scores)
            head_output = torch.sum(token_dependent * 1, dim=1)
            head_outputs.append(head_output)

        context_vector = torch.cat(head_outputs, dim=1)
        return context_vector, token_dependent


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(Attention, self).__init__()
        self.multihead_attention = MultiheadAttention(hidden_size, num_heads)

    def forward(self, x, input_lengths, attention_weights=None):
        return self.multihead_attention(x, input_lengths, attention_weights)
