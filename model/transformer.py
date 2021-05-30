import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention

from .transformer_embedding import TransformerEmbedding


class Transformer(nn.Module):
    def __init__(self, src_vocab_num, trg_vocab_num, pad_idx=0, bos_idx=1, eos_idx=2, 
            d_model=512, d_embedding=256, n_head=8, dim_feedforward=2048, num_encoder_layer=10, 
            num_decoder_layer=10, num_mask_layer=4, src_max_len=100, trg_max_len=100, dropout=0.1):

        super(Transformer, self).__init__()

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len

        # Source embedding part
        self.src_embedding = TransformerEmbedding(src_vocab_num, d_model, d_embedding, 
            pad_idx=self.pad_idx, max_len=self.src_max_len, dropout=dropout)
        self.src_output_linear = nn.Linear(d_model, d_embedding)
        self.src_output_norm = nn.LayerNorm(d_embedding, eps=1e-12)
        self.src_output_linear2 = nn.Linear(d_embedding, src_vocab_num)
        
        # Target embedding part
        self.trg_embedding = TransformerEmbedding(trg_vocab_num, d_model, d_embedding,
            pad_idx=self.pad_idx, max_len=self.trg_max_len, dropout=dropout)
        self.trg_output_linear = nn.Linear(d_model, d_embedding)
        self.trg_output_norm = nn.LayerNorm(d_embedding, eps=1e-12)
        self.trg_output_linear2 = nn.Linear(d_embedding, trg_vocab_num)
        
        # Transformer
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model, self_attn, dim_feedforward, dropout=dropout) \
                for i in range(num_encoder_layer)])

        if num_mask_layer > 0:
            self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
            self.mask_encoders = nn.ModuleList([
                TransformerEncoderLayer(d_model, self_attn, dim_feedforward, dropout=dropout) \
                    for i in range(num_mask_layer)])
        else:
            self.mask_encoders = None

        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        decoder_mask_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.decoders = nn.ModuleList([
            TransformerDecoderLayer(d_model, self_attn, decoder_mask_attn,
                dim_feedforward, dropout=dropout) for i in range(num_decoder_layer)])

        self.dropout = nn.Dropout(dropout)

    @autocast()
    def forward(self, src_input_sentence, trg_input_sentence, tgt_mask, non_pad_position=None):
        src_key_padding_mask = (src_input_sentence == self.pad_idx)
        tgt_key_padding_mask = (trg_input_sentence == self.pad_idx)

        encoder_out = self.src_embedding(src_input_sentence).transpose(0, 1)
        for encoder in self.encoders:
            encoder_out = encoder(encoder_out, src_key_padding_mask=src_key_padding_mask)

        decoder_out = self.trg_embedding(trg_input_sentence).transpose(0, 1)
        for decoder in self.decoders:
            decoder_out = decoder(decoder_out, encoder_out, tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        decoder_out = decoder_out.transpose(0, 1).contiguous()
        if non_pad_position is not None:
            decoder_out = decoder_out[non_pad_position]

        decoder_out = self.trg_output_norm(self.dropout(F.gelu(self.trg_output_linear(decoder_out))))
        decoder_out = self.trg_output_linear2(decoder_out)
        return decoder_out

    @autocast()
    def reconstruct_predict(self, src_sentence, masked_position=None):
        encoder_out = self.src_embedding(src_sentence).transpose(0, 1)
        src_key_padding_mask = src_sentence == self.pad_idx

        for encoder in self.encoders:
            encoder_out = encoder(encoder_out, src_key_padding_mask=src_key_padding_mask)

        if self.mask_encoders:
            for mask_encoder in self.mask_encoders:
                encoder_out = mask_encoder(encoder_out, src_key_padding_mask=src_key_padding_mask)

        encoder_out = encoder_out.transpose(0, 1).contiguous()
        if masked_position is not None:
            encoder_out = encoder_out[masked_position]
        
        encoder_out = self.src_output_norm(self.dropout(F.gelu(self.src_output_linear(encoder_out))))
        encoder_out = self.src_output_linear2(encoder_out)
        return encoder_out

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = torch.tril(torch.ones(sz, sz, dtype=torch.float, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    @autocast()
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, mask_attn, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.multihead_attn = mask_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    @autocast()
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
