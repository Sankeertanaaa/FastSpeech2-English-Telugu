
import torch
import torch.nn as nn
import numpy as np

import transformer.Constants as Constants
from .Layers import FFTBlock
from text import load_phone_map


def normalize_language(language):
    """Convert language codes and names to standard vocabulary names."""

    language_aliases = {
        "en": "English",
        "english": "English",
        "te": "Telugu",
        "te-in": "Telugu",
        "telugu": "Telugu",
    }

    normalized = language_aliases.get(str(language).lower())

    if normalized is None:
        raise ValueError(
            f"Unsupported language: {language}. "
            "Supported languages: English and Telugu."
        )

    return normalized


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """Create sinusoidal positional encoding."""

    def cal_angle(position, hid_idx):
        return position / np.power(
            10000, 2 * (hid_idx // 2) / d_hid
        )

    def get_posi_angle_vec(position):
        return [
            cal_angle(position, hid_j)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [
            get_posi_angle_vec(pos_i)
            for pos_i in range(n_position)
        ]
    )

    sinusoid_table[:, 0::2] = np.sin(
        sinusoid_table[:, 0::2]
    )
    sinusoid_table[:, 1::2] = np.cos(
        sinusoid_table[:, 1::2]
    )

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    """FastSpeech 2 encoder."""

    def __init__(self, config):
        super(Encoder, self).__init__()

        self.language = normalize_language(
            config.get("language", "English")
        )

        vocabulary = load_phone_map(self.language)
        n_src_vocab = len(vocabulary)

        self.vocab_size = n_src_vocab
        self.max_seq_len = config["max_seq_len"]

        n_position = self.max_seq_len + 1

        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]

        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )

        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab,
            d_word_vec,
            padding_idx=Constants.PAD,
        )

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(
                n_position,
                d_word_vec,
                padding_idx=Constants.PAD,
            ).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model,
                    n_head,
                    d_k,
                    d_v,
                    d_inner,
                    kernel_size,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask, return_attns=False):

        batch_size, max_len = src_seq.shape

        slf_attn_mask = mask.unsqueeze(1).expand(
            -1, max_len, -1
        )

        if not self.training and max_len > self.max_seq_len:

            position_enc = get_sinusoid_encoding_table(
                max_len,
                self.d_model,
                padding_idx=Constants.PAD,
            ).to(src_seq.device)

            enc_output = (
                self.src_word_emb(src_seq)
                + position_enc.unsqueeze(0).expand(
                    batch_size, -1, -1
                )
            )

        else:

            enc_output = (
                self.src_word_emb(src_seq)
                + self.position_enc[:, :max_len, :].expand(
                    batch_size, -1, -1
                )
            )

        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(
                enc_output,
                mask=mask,
                slf_attn_mask=slf_attn_mask,
            )

        return enc_output


class Decoder(nn.Module):
    """FastSpeech 2 decoder."""

    def __init__(self, config):
        super(Decoder, self).__init__()

        n_position = config["max_seq_len"] + 1

        d_word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]

        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )

        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(
                n_position,
                d_word_vec,
            ).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model,
                    n_head,
                    d_k,
                    d_v,
                    d_inner,
                    kernel_size,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, mask, return_attns=False):

        batch_size, max_len = enc_seq.shape[:2]

        if not self.training and max_len > self.max_seq_len:

            slf_attn_mask = mask.unsqueeze(1).expand(
                -1, max_len, -1
            )

            position_enc = get_sinusoid_encoding_table(
                max_len,
                self.d_model,
            ).to(enc_seq.device)

            dec_output = (
                enc_seq
                + position_enc.unsqueeze(0).expand(
                    batch_size, -1, -1
                )
            )

        else:

            max_len = min(max_len, self.max_seq_len)

            dec_output = (
                enc_seq[:, :max_len, :]
                + self.position_enc[:, :max_len, :].expand(
                    batch_size, -1, -1
                )
            )

            mask = mask[:, :max_len]

            slf_attn_mask = mask.unsqueeze(1).expand(
                -1, max_len, -1
            )

        for dec_layer in self.layer_stack:
            dec_output, _ = dec_layer(
                dec_output,
                mask=mask,
                slf_attn_mask=slf_attn_mask,
            )

        return dec_output, mask
