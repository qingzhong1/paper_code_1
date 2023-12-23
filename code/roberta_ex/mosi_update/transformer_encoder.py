from torch import nn
import torch
from multihead_attention import MultiheadAttention
import math
import numpy as np
from enum import Enum
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
        return self.weight.to(x.device.type) * x + self.bias.to(x.device.type)

def LayerNorm(embedding_dim):
    m = BertLayerNorm(hidden_size=embedding_dim)
    return m
class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim=768, num_heads=4, attn_dropout=0.1,
                 activate_name='relu',
                 relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask
        self.activate_name=activate_name
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        # Memory and Compound control
        self.mem_proj = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.att_proj = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid()
        )

        # Dense Layer
        if self.activate_name=='relu' or self.activate_name=='glue':
            self.fc1 = nn.Linear(self.embed_dim, 4 * self.embed_dim)  # The "Add & Norm" part in the paper
            self.fc2 = nn.Linear(4 * self.embed_dim, self.embed_dim)
        elif self.activate_name=='swiglu':
            self.w1=nn.Linear(self.embed_dim,(self.embed_dim*8)//3,bias=False)
            self.w2=nn.Linear((self.embed_dim*8)//3,self.embed_dim,bias=False)
            self.w3=nn.Linear(self.embed_dim,(self.embed_dim*8)//3,bias=False)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])
        self.activate_name=activate_name
    def forward(self, mode_Q,mode_K=None,attention_Q=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
            ctc_vec (Tensor): The control vector generated from DIV encoder
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual =mode_Q
        x = self.maybe_layer_norm(0, mode_Q, before=True)
        if mode_K is None:
            x= self.self_attn(query=x, key=x, value=x, add_mask=attention_Q)
        else:
            if mode_K.shape[-1]!=mode_Q.shape[-1]:
                mode_K=nn.Linear(mode_K.shape[-1],mode_Q.shape[-1])(mode_K)
            x_k = self.maybe_layer_norm(0, mode_K, before=True)
            x = self.self_attn(query=x, key=x_k, value=x_k, add_mask=attention_Q)
        x = nn.functional.dropout(x, p=self.res_dropout, training=self.training)

        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        if self.activate_name=='relu':
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
            x = nn.functional.dropout(x, p=self.res_dropout, training=self.training)
        elif self.activate_name=='glue':
            x=nn.functional.gelu(self.fc1(x))
            x = nn.functional.dropout(x, p=self.relu_dropout, training=self.training)
            x = self.fc2(x)
            x = nn.functional.dropout(x, p=self.res_dropout, training=self.training)
        elif self.activate_name=='swiglu':
            x=self.w2(nn.functional.silu(self.w1(x))*self.w3(x))
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

'''model=TransformerEncoderLayer(embed_dim=768,num_heads=12)
Q=torch.randn(size=(20,10,768))
K=torch.randn(size=(20,10,47))
mask=torch.randn(size=(20,10))
output=model(Q,K,mask)
print(output.shape)'''


class GatedTransformer(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0,attn_mask=False):
        super().__init__()
        # num_heads=5,embed_dim=40,layers=5
        # attn_dropout=0.1,relu_dropout=0.1,res_dropout=0.1,embed_dropout=0.25,div_dropout=0.1
        self.dropout = embed_dropout  # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.attn_mask = attn_mask

        # a pair of transformers plus a domain-invariant encoder
        self.l2other_layers = nn.ModuleList([])
        self.other2l_layers = nn.ModuleList([])
        self.div_encoders = nn.ModuleList([])

        for layer in range(layers):
            l2other_new = TransformerEncoderLayer(embed_dim,
                                                  num_heads=num_heads,
                                                  attn_dropout=attn_dropout,
                                                  relu_dropout=relu_dropout,
                                                  res_dropout=res_dropout,
                                                  attn_mask=attn_mask)
            self.l2other_layers.append(l2other_new)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)
    def position(self,x_in,x_embedding):
        self.embed_scale=math.sqrt(x_embedding)
        x = self.embed_scale * x_in
        position=SinusoidalPositionalEmbedding(embedding_dim=x_embedding)(x[:,:,0])
        if x_embedding%2==0:
            return position
        else:
            return position[:,:,:-1]
        return position
    def forward(self, seq_l, seq_other,lengths=None, mask=None):

        """Forward 2 input modals thorugh the DIVencoder and Trnasformer
        Args:
            input_l (FloatTensor): Representative tensor of the language modal
            input_other (FloatTensor): Representative tensor of the other modal
        """
        '''seq_l=self.position(seq_l,seq_l.shape[-1])+seq_l'''
        seq_l=BertLayerNorm(seq_l.shape[-1])(seq_l)
        seq_l= nn.Dropout(0.1)(seq_l)
        '''position_other=self.position(seq_other,seq_other.shape[-1])
        seq_other=position_other+seq_other'''
        seq_other=BertLayerNorm(seq_other.shape[-1])(seq_other)
        seq_other=nn.Dropout(0.1)(seq_other)
        # output all shared encoding to train the discriminator
        # enc_l_all = []
        # enc_other_all = []

        # outputs of all discriminators in every layer
        input_l, input_other = seq_l, seq_other

        # add residual connection
        # resl_all = []
        # resother_all = []

        for layer_i, trans_l2other in enumerate(self.l2other_layers):


            # project language to other modals
            l2other = trans_l2other(input_l,input_other,mask)
            input_l, input_other =l2other,l2other

        return l2other
def get_mask(feature,lengths):
    assert feature.shape[0]==lengths.shape[0]
    uttrace=feature.shape[1]
    mask_maxtrix=np.zeros(shape=(feature.shape[0],feature.shape[1]))
    for i in range(lengths.shape[0]):
        if lengths[i]<=uttrace:
            mask_maxtrix[i,:lengths[i]]=1
        if lengths[i]>uttrace:
            mask_maxtrix[i,:uttrace]=1
    return torch.Tensor(mask_maxtrix)

# Code adapted from the fairseq repo.

def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    device = tensor.get_device()
    buf_name = f'range_buf_{device}'
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, tensor.new())
    setattr(make_positions, buf_name, getattr(make_positions, buf_name).type_as(tensor))
    if getattr(make_positions, buf_name).numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=getattr(make_positions, buf_name))
    mask = tensor.ne(padding_idx)
    positions = getattr(make_positions, buf_name)[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    new_tensor = tensor.clone()
    return new_tensor.masked_scatter_(mask, positions[mask]).long()


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = dict()  # device --> actual weight; due to nn.DataParallel :-(
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb_c1 = math.log(10000) / (half_dim - 1)
        emb_c2 = torch.arange(embedding_dim, dtype=torch.int32)

        emb = torch.exp((emb_c2 // 2).to(torch.float) * -emb_c1)  # (embedding_dim,)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(
            0)  # (num_emb, embedding_dim)

        # assign sinusoidal positional embedding to correct positions
        emb[:, emb_c2 % 2 == 0] = torch.sin(emb[:, emb_c2 % 2 == 0])
        emb[:, emb_c2 % 2 == 1] = torch.cos(emb[:, emb_c2 % 2 == 1])

        # emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1) # (num_emb, half_dim*2)

        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.get_device()
        if device not in self.weights or max_pos > self.weights[device].size(0):
            # recompute/expand embeddings if needed
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights[device] = self.weights[device].type_as(self._float_tensor)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        self.weights[device]=self.weights[device].to(positions.device.type)
        return self.weights[device].index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


'''a = torch.randn(size=(20, 10))
model = SinusoidalPositionalEmbedding(embedding_dim=768)
print(model(a).shape)
model=GatedTransformer(embed_dim=768,num_heads=12,layers=12,attn_dropout=0.1,relu_dropout=0.1,res_dropout=0.1,embed_dropout=0.1)
Q=torch.randn(size=(20,10,768))
K=torch.randn(size=(20,10,47))
mask=torch.randn(size=(20,10))
output=model(Q,K,mask=mask)
print(output.shape)'''
class GatedTransformer_fusion(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0,
                 relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0,attn_mask=False,o_layer=4,l_layer=4,
                 o_heads=12,l_heads=12,o_e=False,l_e=False,mul_lay=4,
                 m_head=12,activate='swiglu'):
        super().__init__()
        # num_heads=5,embed_dim=40,layers=5
        # attn_dropout=0.1,relu_dropout=0.1,res_dropout=0.1,embed_dropout=0.25,div_dropout=0.1
        self.dropout = embed_dropout  # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.attn_mask = attn_mask

        # a pair of transformers plus a domain-invariant encoder
        self.l2other_layers = nn.ModuleList([])
        self.other2l_layers = nn.ModuleList([])
        self.div_encoders = nn.ModuleList([])
        self.l_self_layers=nn.ModuleList([])
        self.o_self_layers=nn.ModuleList([])
        self.mul_self=nn.ModuleList([])
        self.o_layer=o_layer
        self.l_layer=l_layer
        self.o_e=o_e
        self.l_e=l_e

        self.mul_lay=mul_lay
        for _ in range(self.o_layer):
            o_layer = TransformerEncoderLayer(embed_dim, num_heads=o_heads, attn_dropout=attn_dropout,
                                             relu_dropout=relu_dropout,activate_name=activate)
            self.o_self_layers.append(o_layer)
        for _ in range(self.l_layer):
            l_layer=TransformerEncoderLayer(embed_dim,num_heads=l_heads,attn_dropout=attn_dropout,
                                            relu_dropout=relu_dropout,activate_name=activate)
            self.l_self_layers.append(l_layer)
        for _ in range(self.mul_lay):
            mul_layer = TransformerEncoderLayer(embed_dim, num_heads=m_head, attn_dropout=attn_dropout,
                                               relu_dropout=relu_dropout,activate_name=activate)
            self.mul_self.append(mul_layer)
        for layer in range(layers):
            l2other_new = TransformerEncoderLayer(embed_dim,
                                                  num_heads=num_heads,
                                                  attn_dropout=attn_dropout,
                                                  relu_dropout=relu_dropout,
                                                  res_dropout=res_dropout,
                                                  attn_mask=attn_mask,activate_name=activate)
            self.l2other_layers.append(l2other_new)
            self.other2l_layers.append(l2other_new)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)
    def position(self,x_in,x_embedding):
        self.embed_scale=math.sqrt(x_embedding)
        x = self.embed_scale * x_in
        position=SinusoidalPositionalEmbedding(embedding_dim=x_embedding)(x[:,:,0])
        if x_embedding%2==0:
            return position
        else:
            return position[:,:,:-1]
        return position
    def forward(self, seq_l, seq_other, mask_l=None,mask_o=None):

        """Forward 2 input modals thorugh the DIVencoder and Trnasformer
        Args:
            input_l (FloatTensor): Representative tensor of the language modal
            input_other (FloatTensor): Representative tensor of the other modal
        """
        '''seq_l=self.position(seq_l,seq_l.shape[-1])+seq_l'''
        ###位置编码
        if self.l_e:
            seq_l=self.position(seq_l,seq_l.shape[-1])+seq_l
        seq_l=BertLayerNorm(seq_l.shape[-1])(seq_l)
        seq_l= nn.Dropout(0.1)(seq_l)
        if self.o_e:
            seq_other=self.position(seq_other,seq_other.shape[-1])
        '''position_other=self.position(seq_other,seq_other.shape[-1])
        seq_other=position_other+seq_other'''
        seq_other=BertLayerNorm(seq_other.shape[-1])(seq_other)
        seq_other=nn.Dropout(0.1)(seq_other)
        # output all shared encoding to train the discriminator
        # enc_l_all = []
        # enc_other_all = []

        # outputs of all discriminators in every layer
        input_l, input_other = seq_l, seq_other

        # add residual connection
        # resl_all = []
        # resother_all = []
        for index in range(self.l_layer):
            input_l=self.l_self_layers[index](mode_Q=input_l,attention_Q=mask_l)
        for index in range(self.o_layer):
            input_other=self.o_self_layers[index](mode_Q=input_other,attention_Q=mask_o)
        for _, (trans_l2other,trans_other2l) in enumerate(zip(self.l2other_layers,self.other2l_layers)):
            # project language to other modals
            input_l,input_other= trans_l2other(input_l,input_other,mask_l),trans_other2l(input_other,input_l,mask_o)
        total_feature=torch.concat([input_l,input_other],dim=1).reshape(input_l.shape[0],-1,input_other.shape[-1])
        mask=torch.concat([mask_l.to(total_feature.device.type),mask_o.to(total_feature.device.type)],dim=1).reshape(mask_l.shape[0],-1)
        for index in range(self.mul_lay):
            total_feature=self.mul_self[index](mode_Q=total_feature,attention_Q=mask)
        return total_feature,mask

class PoolingStrategy(str, Enum):
    cls = 'cls'
    last_mean = 'last_mean'
    first_last_mean = 'first_last_mean'
    embedding_last_mean = 'embedding_last_mean'
    last_weighted = 'last_weighted'

def mean_pooling(hidden_state, attention_mask=None):
    if attention_mask is None:
        return torch.mean(hidden_state, dim=1)
    attention_mask = attention_mask.float()
    ###对应每个句子的每个token的同一维度相加除以句子的token数字
    return torch.sum(hidden_state * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=-1, keepdim=True)
def max_pooling(hidden_state,attention_mask=None):
    if attention_mask is None:
        return torch.max(hidden_state,dim=1)
    attention_mask=attention_mask.float()
    return torch.max(hidden_state*attention_mask.unsqueeze(-1),dim=1)[0]
def pool_weight(embeddings,attention_mask=None):
    weights = (torch.arange(embeddings.shape[1], device=embeddings.device) + 1).float()
    if attention_mask==None:
        embeddings = embeddings * weights.unsqueeze(0).unsqueeze(-1)
    else:
        embeddings=embeddings * attention_mask.unsqueeze(-1).float() * weights.unsqueeze(0).unsqueeze(-1)
    embeddings = torch.sum(embeddings, dim=1) / torch.sum(weights * attention_mask, dim=-1, keepdim=True)
    return embeddings
def pool_attention(hidden_state,attention_mask=None):
    att=torch.matmul(hidden_state,hidden_state.transpose(-1,-2))
    if attention_mask:
        attention_mask=1-attention_mask
        attention_mask=torch.where(attention_mask!=1,attention_mask,float='-inf')
        weight=att+attention_mask.unsqueeze(dim=-1)
    else:
        weight=att
    weight=torch.sum(weight,dim=-1)
    weight=torch.nn.functional.softmax(weight,dim=-1).unsqueeze(-1)
    return torch.sum(weight*hidden_state,dim=1)
    

def token2sentence(token_embedding,method='cls',mask=None):
    if method=='cls':
        return token_embedding[:,1,:]
    elif method=='mean':
        return mean_pooling(hidden_state=token_embedding,attention_mask=mask)
    elif method=='max':
        return max_pooling(hidden_state=token_embedding,attention_mask=mask)
    elif method=='weight_sum':
        return pool_weight(embeddings=token_embedding,attention_mask=mask)
    #elif method=='attention':
        

