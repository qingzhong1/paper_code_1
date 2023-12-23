from __future__ import absolute_import, division, print_function, unicode_literals
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""
from pytorch_pretrained_bert.file_utils import cached_path
import copy
import json
import logging
import math
from torch.autograd import Variable
from transformer_encoder import GatedTransformer,GatedTransformer_main
import os
import shutil
import torch.nn.functional as F
import tarfile
import tempfile
import sys
from io import open
from transformers import BertModel, BertConfig
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from utils import *

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
TF_WEIGHTS_NAME = 'model.ckpt'


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
              "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def bi_modal_attention(x, y):
    m1 = torch.matmul(x, y.transpose(-1, -2))
    n1 = nn.Softmax(dim=-1)(m1)
    o1 = torch.matmul(n1, y)
    a1 = torch.mul(o1, x)

    m2 = torch.matmul(y, x.transpose(-1, -2))
    n2 = nn.Softmax(dim=-1)(m2)
    o2 = torch.matmul(n2, x)
    a2 = torch.mul(o2, y)
    return a1, a2


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class LanguageEmbeddingLayer(nn.Module):
    """Embed input text with "Bert"
    """
    def __init__(self):
        super(LanguageEmbeddingLayer, self).__init__()
        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)

    def forward(self, bert_sent, bert_sent_type, bert_sent_mask):
        bert_output = self.bertmodel(input_ids=bert_sent,
                                attention_mask=bert_sent_mask,
                                token_type_ids=bert_sent_type)
        encoder_lastoutput = bert_output[0]

            # masked mean
            # masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            # mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
            # output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
        return encoder_lastoutput


class BertFinetun(nn.Module):
    def __init__(self,hidden_dropout_prob,fusion_dim):
        super(BertFinetun, self).__init__()
        self.proj_t = nn.Conv1d(768, 30, kernel_size=1, padding=0, bias=False)
        # self.proj_a = nn.Conv1d(5,30, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(fusion_dim, 30, kernel_size=1, padding=0, bias=False)
        self.activation = nn.ReLU()
        self.audio_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.text_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.audio_weight_1.data.fill_(1)
        self.text_weight_1.data.fill_(1)
        self.bias.data.fill_(0)
        self.dropout1 = nn.Dropout(0.3)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm1 = BertLayerNorm(768)

    def forward(self, hidden_states, audio_data, attention_mask):
        attention_mask = attention_mask.squeeze(1)
        attention_mask_ = attention_mask.permute(0, 2, 1)
        text_data = hidden_states
        text_data = text_data.transpose(1, 2)
        text_data = self.proj_t(text_data)
        text_data = text_data.transpose(1, 2)
        text_data_1 = text_data.reshape(-1).cpu().detach().numpy()
        weights = np.sqrt(np.linalg.norm(text_data_1, ord=2))
        text_data = text_data / weights

        audio_data = audio_data.transpose(1, 2)
        audio_data = self.proj_a(audio_data)
        audio_data = audio_data.transpose(1, 2)

        text_att = torch.matmul(text_data, text_data.transpose(-1, -2))
        text_att1 = self.activation(text_att)

        audio_att = torch.matmul(audio_data, audio_data.transpose(-1, -2))
        audio_att = self.activation(audio_att)

        audio_weight_1 = self.audio_weight_1
        text_weight_1 = self.text_weight_1
        bias = self.bias

        fusion_att = text_weight_1 * text_att1 + audio_weight_1 * audio_att + bias
        fusion_att1 = self.activation(fusion_att)
        fusion_att = fusion_att + attention_mask + attention_mask_
        fusion_att = nn.Softmax(dim=-1)(fusion_att)
        fusion_att = self.dropout1(fusion_att)

        fusion_data = torch.matmul(fusion_att, hidden_states)
        fusion_data = fusion_data + hidden_states

        hidden_states_new = self.dense(fusion_data)
        hidden_states_new = self.dropout(hidden_states_new)
        hidden_states_new = self.LayerNorm1(hidden_states_new)
        hidden_states_new = hidden_states_new[:, 0]
        return hidden_states_new, text_att1, fusion_att1
def init_bert_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
class BertForSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, hidden_dropout_prob,hidden_size, num_labels,fusion_dim):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = LanguageEmbeddingLayer()
        self.BertFinetun = BertFinetun(hidden_dropout_prob,fusion_dim)
        self.dropout = nn.Dropout(hidden_dropout_prob)#拿出来CLS的表示dimension = 768
        self.classifier = nn.Linear(hidden_size, 1)
    def forward(self, input_ids, all_audio_data,token_type_ids=None, attention_mask=None, labels=None):
        encoder_lastoutput= self.bert(input_ids,token_type_ids, attention_mask)
        pooled_output = self.dropout(encoder_lastoutput)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        pooled_output,text_att,fusion_att = self.BertFinetun(pooled_output,all_audio_data,extended_attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss = 0.5 * (logits.view(-1) - labels) ** 2
            return loss,logits
        else:
            return logits,text_att,fusion_att
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'



class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dims, hidden_dims, dropouts, output_dim, rank):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, hidden dims of the sub-networks
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(LMF, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]

        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]

        self.output_dim = output_dim
        self.rank = rank

        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.post_fusion_prob = dropouts[2]

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        # self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.audio_factor = nn.Parameter(torch.Tensor(self.rank, self.audio_hidden + 1, self.output_dim))
        self.video_factor = nn.Parameter(torch.Tensor(self.rank, self.video_hidden + 1, self.output_dim))
        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        nn.init.xavier_normal(self.audio_factor)
        nn.init.xavier_normal(self.video_factor)
        nn.init.xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, audio_x, video_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        batch_size = audio_x.data.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1), requires_grad=False), audio_x), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1), requires_grad=False), video_x), dim=1)

        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_zy = fusion_audio * fusion_video

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        return output
def A_V(input_aduio,input_vision,output_dim):
    uttrace=input_aduio.shape[1]
    input_aduio=input_aduio.permute([1,0,2])
    input_vision=input_vision.permute([1,0,2])
    fusion=[]
    for i in range(uttrace):
        input_aduio_item=input_aduio[i,:,:]
        input_vision_item=input_vision[i,:,:]
        fusion_item=LMF(input_dims=[input_aduio_item.shape[-1],input_vision_item.shape[-1]],hidden_dims=[input_aduio_item.shape[-1],input_vision_item.shape[-1]],rank=max(input_aduio_item.shape[-1],input_vision_item.shape[-1]),output_dim=output_dim,dropouts=[0.1,0.1,0.1])(input_aduio_item,input_vision_item)
        fusion.append(fusion_item)
    fusion=torch.stack(fusion,dim=0)
    return fusion.permute([1,0,2])




class BertForSequenceClassification_three(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, hidden_dropout_prob,hidden_size, num_labels,fusion_dim):
        super(BertForSequenceClassification_three, self).__init__()
        self.num_labels = num_labels
        self.bert = LanguageEmbeddingLayer()
        self.BertFinetun = BertFinetun(hidden_dropout_prob,fusion_dim=fusion_dim)
        self.dropout = nn.Dropout(hidden_dropout_prob)#拿出来CLS的表示dimension = 768
        self.classifier = nn.Linear(hidden_size, 1)
        self.fusion_dim=fusion_dim
    def forward(self, input_ids, all_audio_data,all_vision_data,token_type_ids=None, attention_mask=None, labels=None):
        encoder_lastoutput= self.bert(input_ids,token_type_ids, attention_mask)
        pooled_output = self.dropout(encoder_lastoutput)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        fusion=A_V(all_audio_data,all_vision_data,self.fusion_dim)
        pooled_output,text_att,fusion_att = self.BertFinetun(pooled_output,fusion,extended_attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss = 0.5 * (logits.view(-1) - labels) ** 2
            return loss,logits
        else:
            return logits,text_att,fusion_att

def length_mask(length,maxlen):
    mask_new=[]
    for i in range(len(length)):
        if length[i]<maxlen:
            mask_new_item=[1 for j in range(length[i])]+[0 for j in range(maxlen-length[i])]
            mask_new.append(mask_new_item)
        else:
            mask_new_item=[1 for j in range(maxlen)]
            mask_new.append(mask_new_item)
    mask_new=torch.tensor(mask_new)
    return mask_new
class Bert_LMF_Tranformer(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, hidden_dropout_prob,hidden_size, num_labels,fusion_dim):
        super(Bert_LMF_Tranformer, self).__init__()
        self.num_labels = num_labels
        self.bert = LanguageEmbeddingLayer()
        self.BertFinetun = BertFinetun(hidden_dropout_prob,fusion_dim=fusion_dim)
        self.tranformer_T2AV=GatedTransformer(embed_dim=hidden_size,num_heads=12,layers=5,attn_dropout=0.1,relu_dropout=0.1,res_dropout=0.1,embed_dropout=0.1)
        self.tranformer_AV2T = GatedTransformer(embed_dim=fusion_dim, num_heads=5, layers=5, attn_dropout=0.1,
                                                relu_dropout=0.1, res_dropout=0.1, embed_dropout=0.1)
        self.dropout = nn.Dropout(hidden_dropout_prob)#拿出来CLS的表示dimension = 768
        self.classifier = nn.Linear(hidden_size, 1)
        self.fusion_dim=fusion_dim
    def forward(self, input_ids, all_audio_data,all_vision_data,token_type_ids=None, attention_mask=None,V_A_mask=None, labels=None):
        encoder_lastoutput= self.bert(input_ids,token_type_ids, attention_mask)
        pooled_output = self.dropout(encoder_lastoutput)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        fusion=A_V(all_vision_data,all_audio_data,self.fusion_dim)
        T2VA=self.tranformer_T2AV(pooled_output,fusion)
        VA_T=self.tranformer_AV2T(fusion,pooled_output)
        pooled_output,text_att,fusion_att = self.BertFinetun(T2VA,VA_T,extended_attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss = 0.5 * (logits.view(-1) - labels) ** 2
            return loss,logits
        else:
            return logits,text_att,fusion_att


class LightAttentiveAggregation(torch.nn.Module):
    def __init__(self, audio_output_size=768):
        super().__init__()
        ##BiGRU
        self.audio_output_size = audio_output_size
        self.bigru = torch.nn.GRU(input_size=768, hidden_size=audio_output_size, num_layers=2, bias=True,
                                  batch_first=True, bidirectional=True)

        ##Aggregation / Attention Module check paper citation 40
        self.agg_1 = torch.nn.Linear(in_features=audio_output_size, out_features=audio_output_size, bias=True)

        self.agg_2 = torch.nn.Linear(in_features=audio_output_size, out_features=1, bias=True)
    def forward(self, fp_out):
        gru_output_sequence, _ = self.bigru(fp_out)
        gru_output_sequence = gru_output_sequence.view(-1, gru_output_sequence.shape[1], 2, self.audio_output_size)
        p1_gru_output_sequence, p2_gru_output_sequence = gru_output_sequence[:, :, 0, :], gru_output_sequence[:, :, 1,
                                                                                          :]
        u_1 = torch.sigmoid(self.agg_1(p1_gru_output_sequence))
        u_2 = torch.sigmoid(self.agg_1(p2_gru_output_sequence))
        alph_1 = torch.softmax(self.agg_2(u_1), 1)
        alph_2 = torch.softmax(self.agg_2(u_2), 1)
        ca_1 = torch.bmm(alph_1.transpose(1, 2), p1_gru_output_sequence)
        ca_2 = torch.bmm(alph_2.transpose(1, 2), p2_gru_output_sequence)
        ca = torch.concat((ca_1, ca_2), -1)
        return ca
class BertFinetun_1(nn.Module):
    def __init__(self,hidden_dropout_prob,fusion_dim):
        super(BertFinetun_1, self).__init__()
        self.proj_t = nn.Conv1d(768, 30, kernel_size=1, padding=0, bias=False)
        # self.proj_a = nn.Conv1d(5,30, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(768, 30, kernel_size=1, padding=0, bias=False)
        self.activation = nn.ReLU()
        self.audio_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.text_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.audio_weight_1.data.fill_(1)
        self.text_weight_1.data.fill_(1)
        self.bias.data.fill_(0)
        self.dropout1 = nn.Dropout(0.3)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm1 = BertLayerNorm(768)
        self.tranformer_LVA=GatedTransformer_main(embed_dim=768,num_heads=12,layers=5,attn_dropout=0.1,relu_dropout=0.1, res_dropout=0.1, embed_dropout=0.1)
    def forward(self, hidden_states, audio_data, attention_mask):
        attention_mask = attention_mask.squeeze(1)
        attention_mask_ = attention_mask.permute(0, 2, 1)
        audio_data = self.tranformer_LVA(hidden_states, audio_data)
        text_data = hidden_states
        text_data = text_data.transpose(1, 2)
        text_data = self.proj_t(text_data)
        text_data = text_data.transpose(1, 2)
        text_data_1 = text_data.reshape(-1).cpu().detach().numpy()
        weights = np.sqrt(np.linalg.norm(text_data_1, ord=2))
        text_data = text_data / weights


        audio_data = audio_data.transpose(1, 2)
        audio_data = self.proj_a(audio_data)
        audio_data = audio_data.transpose(1, 2)

        text_att = torch.matmul(text_data, text_data.transpose(-1, -2))
        text_att1 = self.activation(text_att)

        audio_att = torch.matmul(audio_data, audio_data.transpose(-1, -2))
        audio_att = self.activation(audio_att)

        audio_weight_1 = self.audio_weight_1
        text_weight_1 = self.text_weight_1
        bias = self.bias

        fusion_att = text_weight_1 * text_att1 + audio_weight_1 * audio_att + bias
        fusion_att1 = self.activation(fusion_att)
        fusion_att = fusion_att + attention_mask + attention_mask_
        fusion_att = nn.Softmax(dim=-1)(fusion_att)
        fusion_att = self.dropout1(fusion_att)

        fusion_data = torch.matmul(fusion_att, hidden_states)
        fusion_data = fusion_data + hidden_states

        hidden_states_new = self.dense(fusion_data)
        hidden_states_new = self.dropout(hidden_states_new)
        hidden_states_new = self.LayerNorm1(hidden_states_new)
        return hidden_states_new, text_att1, fusion_att1
from transformers import RobertaModel
class Roberta(nn.Module):
    def __init__(self, output_size,audio_size,vision_size):
        super(Roberta, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.la= torch.nn.GRU(input_size=audio_size, hidden_size=output_size, num_layers=2, bias=True,
                                  batch_first=True, bidirectional=True)
        self.lv =torch.nn.GRU(input_size=vision_size, hidden_size=output_size, num_layers=2, bias=True,
                                  batch_first=True, bidirectional=True)
        self.tranformer_VA = GatedTransformer(embed_dim=output_size*2, num_heads=5, layers=5, attn_dropout=0.1,
                                                relu_dropout=0.1, res_dropout=0.1, embed_dropout=0.1)
        self.bertfinue=BertFinetun_1(hidden_dropout_prob=0.1,fusion_dim=output_size*2)
        self.laa=LightAttentiveAggregation()
        self.dropout_1=nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)
        self.classifier=nn.Linear(768*2,1)
    def forward(self, input_ids, all_audio_data,all_vision_data,token_type_ids=None, attention_mask=None,labels=None):
        output_text= self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        audio_data,_=self.la(all_audio_data)
        vision_data,_=self.lv(all_vision_data)

        V_A=self.tranformer_VA(vision_data,audio_data)
        #pooled_output = self.dropout_1(output_text)
        pooled_output = output_text['last_hidden_state']
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        hidden_states_new, text_att1, fusion_att1=self.bertfinue(pooled_output,V_A,extended_attention_mask)
        hidden_states_new=self.laa(hidden_states_new)
        #hidden_states_new=torch.squeeze(hidden_states_new[:,0])
        hidden_states_new = self.dropout_2(hidden_states_new)
        logits = self.classifier(hidden_states_new)
        if labels is not None:
            loss = 0.5 * (logits.view(-1) - labels) ** 2
            return loss, logits
        else:
            return logits


class Roberta_two(nn.Module):
    def __init__(self, output_size,audio_size,vision_size):
        super(Roberta_two, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.la= torch.nn.GRU(input_size=audio_size, hidden_size=output_size, num_layers=2, bias=True,
                                  batch_first=True, bidirectional=True)
        self.lv =torch.nn.GRU(input_size=vision_size, hidden_size=output_size, num_layers=2, bias=True,
                                  batch_first=True, bidirectional=True)
        self.tranformer_VA = GatedTransformer(embed_dim=output_size*2, num_heads=5, layers=5, attn_dropout=0.1,
                                                relu_dropout=0.1, res_dropout=0.1, embed_dropout=0.1)
        self.bertfinue=BertFinetun_1(hidden_dropout_prob=0.1,fusion_dim=output_size*2)
        self.laa=LightAttentiveAggregation()
        self.dropout_1=nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)
        self.classifier=nn.Linear(768*2,2)
    def forward(self, input_ids, all_audio_data,all_vision_data,token_type_ids=None, attention_mask=None,labels=None):
        output_text= self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        audio_data,_=self.la(all_audio_data)
        vision_data,_=self.lv(all_vision_data)
        V_A=self.tranformer_VA(vision_data,audio_data)
        #pooled_output = self.dropout_1(output_text)
        pooled_output = output_text['last_hidden_state']
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        hidden_states_new, text_att1, fusion_att1=self.bertfinue(pooled_output,V_A,extended_attention_mask)
        hidden_states_new=self.laa(hidden_states_new)
        #hidden_states_new=torch.squeeze(hidden_states_new[:,0])
        hidden_states_new = self.dropout_2(hidden_states_new)
        logits = self.classifier(hidden_states_new)
        if labels is not None:
            loss = 0.5 * (logits.view(-1) - labels) ** 2
            #loss = abs(logits.view(-1) - labels)
            return loss, logits
        else:
            return logits



class Roberta_seven(nn.Module):
    def __init__(self, output_size,audio_size,vision_size):
        super(Roberta_seven, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.la= torch.nn.GRU(input_size=audio_size, hidden_size=output_size, num_layers=2, bias=True,
                                  batch_first=True, bidirectional=True)
        self.lv =torch.nn.GRU(input_size=vision_size, hidden_size=output_size, num_layers=2, bias=True,
                                  batch_first=True, bidirectional=True)
        self.tranformer_VA = GatedTransformer(embed_dim=output_size*2, num_heads=5, layers=5, attn_dropout=0.1,
                                                relu_dropout=0.1, res_dropout=0.1, embed_dropout=0.1)
        self.bertfinue=BertFinetun_1(hidden_dropout_prob=0.1,fusion_dim=output_size*2)
        self.laa=LightAttentiveAggregation()
        self.dropout_1=nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)
        self.classifier=nn.Linear(768*2,7)
    def forward(self, input_ids, all_audio_data,all_vision_data,token_type_ids=None, attention_mask=None,labels=None):
        output_text= self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        audio_data,_=self.la(all_audio_data)
        vision_data,_=self.lv(all_vision_data)
        V_A=self.tranformer_VA(vision_data,audio_data)
        #pooled_output = self.dropout_1(output_text)
        pooled_output = output_text['last_hidden_state']
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        hidden_states_new, text_att1, fusion_att1=self.bertfinue(pooled_output,V_A,extended_attention_mask)
        hidden_states_new=self.laa(hidden_states_new)
        #hidden_states_new=torch.squeeze(hidden_states_new[:,0])
        hidden_states_new = self.dropout_2(hidden_states_new)
        logits = self.classifier(hidden_states_new)
        logits=nn.functional.sigmoid(logits)
        if labels is not None:
            #loss = 0.5 * (logits.view(-1) - labels) ** 2
            loss = abs(logits.view(-1) - labels)
            return loss, logits
        else:
            return logits


class BertFinetun_noal(nn.Module):
    def __init__(self,hidden_dropout_prob,fusion_dim):
        super(BertFinetun_noal, self).__init__()
        self.proj_t = nn.Conv1d(768, 30, kernel_size=1, padding=0, bias=False)
        # self.proj_a = nn.Conv1d(5,30, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(fusion_dim, 30, kernel_size=1, padding=0, bias=False)
        self.l_a=GatedTransformer(embed_dim=30, num_heads=5, layers=5, attn_dropout=0.1,
                                                relu_dropout=0.1, res_dropout=0.1, embed_dropout=0.1)
        self.activation = nn.ReLU()
        self.audio_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.text_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.audio_weight_1.data.fill_(1)
        self.text_weight_1.data.fill_(1)
        self.bias.data.fill_(0)
        self.dropout1 = nn.Dropout(0.3)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm1 = BertLayerNorm(768)

    def forward(self, hidden_states, audio_data, attention_mask):
        attention_mask = attention_mask.squeeze(1)
        attention_mask_ = attention_mask.permute(0, 2, 1)
        text_data = hidden_states
        text_data = text_data.transpose(1, 2)
        text_data = self.proj_t(text_data)
        text_data = text_data.transpose(1, 2)
        text_data_1 = text_data.reshape(-1).cpu().detach().numpy()
        weights = np.sqrt(np.linalg.norm(text_data_1, ord=2))
        text_data = text_data / weights

        audio_data = audio_data.transpose(1, 2)
        audio_data = self.proj_a(audio_data)
        audio_data = audio_data.transpose(1, 2)
        audio_data=self.l_a(text_data,audio_data)
        text_att = torch.matmul(text_data, text_data.transpose(-1, -2))
        text_att1 = self.activation(text_att)

        audio_att = torch.matmul(audio_data, audio_data.transpose(-1, -2))
        audio_att = self.activation(audio_att)

        audio_weight_1 = self.audio_weight_1
        text_weight_1 = self.text_weight_1
        bias = self.bias

        fusion_att = text_weight_1 * text_att1 + audio_weight_1 * audio_att + bias
        fusion_att1 = self.activation(fusion_att)
        fusion_att = fusion_att + attention_mask + attention_mask_
        fusion_att = nn.Softmax(dim=-1)(fusion_att)
        fusion_att = self.dropout1(fusion_att)

        fusion_data = torch.matmul(fusion_att, hidden_states)
        fusion_data = fusion_data + hidden_states

        hidden_states_new = self.dense(fusion_data)
        hidden_states_new = self.dropout(hidden_states_new)
        hidden_states_new = self.LayerNorm1(hidden_states_new)
        return hidden_states_new, text_att1, fusion_att1
class Roberta_noal(nn.Module):
    def __init__(self, output_size,audio_size,vision_size):
        super(Roberta_noal, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.la= torch.nn.GRU(input_size=audio_size, hidden_size=output_size, num_layers=2, bias=True,
                                  batch_first=True, bidirectional=True)
        self.lv =torch.nn.GRU(input_size=vision_size, hidden_size=output_size, num_layers=2, bias=True,
                                  batch_first=True, bidirectional=True)
        self.tranformer_VA = GatedTransformer(embed_dim=output_size*2, num_heads=5, layers=5, attn_dropout=0.1,
                                                relu_dropout=0.1, res_dropout=0.1, embed_dropout=0.1)
        self.bertfinue=BertFinetun_1(hidden_dropout_prob=0.1,fusion_dim=output_size*2)
        self.laa=LightAttentiveAggregation()
        self.dropout_1=nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)
        self.classifier=nn.Linear(768*2,1)
    def forward(self, input_ids, all_audio_data,all_vision_data,token_type_ids=None, attention_mask=None,labels=None):
        output_text= self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        audio_data,_=self.la(all_audio_data)
        vision_data,_=self.lv(all_vision_data)
        V_A=self.tranformer_VA(vision_data,audio_data)
        #pooled_output = self.dropout_1(output_text)
        pooled_output = output_text['last_hidden_state']
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        hidden_states_new, text_att1, fusion_att1=self.bertfinue(pooled_output,V_A,extended_attention_mask)
        hidden_states_new=self.laa(hidden_states_new)
        #hidden_states_new=torch.squeeze(hidden_states_new[:,0])
        hidden_states_new = self.dropout_2(hidden_states_new)
        logits = self.classifier(hidden_states_new)
        if labels is not None:
            loss = 0.5 * (logits.view(-1) - labels) ** 2
            return loss, logits
        else:
            return logits
class BERT_E(nn.Module):
    def __init__(self,output_size,audio_size,vision_size):
        super(BERT_E, self).__init__()
        self.l1=BertModel.from_pretrained('D:/深度学习/bert_pretraining/')
        self.la = torch.nn.GRU(input_size=audio_size, hidden_size=output_size, num_layers=2, bias=True,
                               batch_first=True, bidirectional=True)
        self.lv = torch.nn.GRU(input_size=vision_size, hidden_size=output_size, num_layers=2, bias=True,
                               batch_first=True, bidirectional=True)
        self.tranformer_VA = GatedTransformer(embed_dim=output_size * 2, num_heads=5, layers=5, attn_dropout=0.1,
                                              relu_dropout=0.1, res_dropout=0.1, embed_dropout=0.1)
        self.bertfinue = BertFinetun_1(hidden_dropout_prob=0.1, fusion_dim=output_size * 2)
        self.laa = LightAttentiveAggregation()
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)
        self.classifier = nn.Linear(768 * 2, 1)

    def forward(self, input_ids, all_audio_data, all_vision_data, token_type_ids=None, attention_mask=None,
                labels=None):
        output_text = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        audio_data, _ = self.la(all_audio_data)
        vision_data, _ = self.lv(all_vision_data)
        V_A = self.tranformer_VA(vision_data, audio_data)
        # pooled_output = self.dropout_1(output_text)
        pooled_output = output_text['last_hidden_state']
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        hidden_states_new, text_att1, fusion_att1 = self.bertfinue(pooled_output, V_A, extended_attention_mask)
        hidden_states_new = self.laa(hidden_states_new)
        # hidden_states_new=torch.squeeze(hidden_states_new[:,0])
        hidden_states_new = self.dropout_2(hidden_states_new)
        logits = self.classifier(hidden_states_new)
        if labels is not None:
            loss = 0.5 * (logits.view(-1) - labels) ** 2
            return loss, logits
        else:
            return logits


''' class Roberta(nn.Module):
    def __init__(self, output_size,audio_size,vision_size):
        super(Roberta, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.la= torch.nn.GRU(input_size=audio_size, hidden_size=output_size, num_layers=2, bias=True,
                                  batch_first=True, bidirectional=True)
        self.lv =torch.nn.GRU(input_size=vision_size, hidden_size=output_size, num_layers=2, bias=True,
                                  batch_first=True, bidirectional=True)
        self.tranformer_VA = GatedTransformer(embed_dim=output_size*2, num_heads=5, layers=5, attn_dropout=0.1,
                                                relu_dropout=0.1, res_dropout=0.1, embed_dropout=0.1)
        self.bertfinue=BertFinetun_1(hidden_dropout_prob=0.1,fusion_dim=output_size*2)
        self.laa=LightAttentiveAggregation()
        self.dropout_1=nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)
        self.classifier=nn.Linear(768*2,1)
    def forward(self, input_ids, all_audio_data,all_vision_data,token_type_ids=None, attention_mask=None,labels=None):
        output_text= self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        audio_data,_=self.la(all_audio_data)
        vision_data,_=self.lv(all_vision_data)

        V_A=self.tranformer_VA(vision_data,audio_data)
        #pooled_output = self.dropout_1(output_text)
        pooled_output = output_text['last_hidden_state']
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        hidden_states_new, text_att1, fusion_att1=self.bertfinue(pooled_output,V_A,extended_attention_mask)
        hidden_states_new=self.laa(hidden_states_new)
        #hidden_states_new=torch.squeeze(hidden_states_new[:,0])
        hidden_states_new = self.dropout_2(hidden_states_new)
        logits = self.classifier(hidden_states_new)
        if labels is not None:
            loss = 0.5 * (logits.view(-1) - labels) ** 2
            return loss, logits
        else:
            return logits'''