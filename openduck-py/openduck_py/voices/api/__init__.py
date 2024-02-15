from typing import List

import torch
import numpy as np
from munch import Munch
import nltk

nltk.download("punkt")

from ..settings import DEVICE


def _length_to_mask(lengths):
    mask = (
        torch.arange(lengths.max())
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i


class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d


# rap remnant
def _scale_segments_to_target_length(
    vector, target_length: int, start_indices: List[int]
) -> List[int]:
    scaled_vector = []
    for i in range(len(start_indices) - 1):
        segment = vector[start_indices[i] : start_indices[i + 1]]
        segment_sum = sum(segment)
        if segment_sum == 0:
            continue

        scaling_factor = target_length / segment_sum

        scaled_segment = [(value * scaling_factor).round() for value in segment]
        scaled_vector.extend(scaled_segment)

    return torch.tensor(scaled_vector)


def inference(
    phonemes,
    model,
    sampler,
    ref_s,
    alpha=0.5,
    beta=0.0,
    diffusion_steps=30,
    embedding_scale=5,
    bps=1.5,
    nlines=1,
    warm_start_index=0,
):
    textcleaner = TextCleaner()
    tokens = textcleaner(phonemes)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(DEVICE).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(DEVICE)
        text_mask = _length_to_mask(input_lengths).to(DEVICE)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(
            noise=torch.randn((1, 256)).unsqueeze(1).to(DEVICE),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_s,  # reference from the same speaker as the embedding
            num_steps=diffusion_steps,
        ).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        start_indices = np.asarray(
            np.linspace(0, int(len(duration[0])), nlines + 1), dtype=int
        ).tolist()
        # seconds_per_beat = bps ** (-1)
        # sample_rate = 24000
        # hop_length = 600  #  300 in config but for some reason off
        # frames_per_beat = seconds_per_beat * sample_rate / hop_length
        # frames_per_line = int(frames_per_beat * 4)

        # duration = _scale_segments_to_target_length(
        #    duration[0], frames_per_line, start_indices
        # )

        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(DEVICE)

        asr_new = torch.zeros_like(en)
        asr_new[:, :, 0] = en[:, :, 0]
        asr_new[:, :, 1:] = en[:, :, 0:-1]
        en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = t_en @ pred_aln_trg.unsqueeze(0).to(DEVICE)
        asr_new = torch.zeros_like(asr)
        asr_new[:, :, 0] = asr[:, :, 0]
        asr_new[:, :, 1:] = asr[:, :, 0:-1]
        asr = asr_new

        hop_length = 600
        sample_rate = 24000
        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        warm_start_time = int(pred_dur[:warm_start_index].sum() * hop_length)

    return (
        out.squeeze().cpu().numpy()[..., warm_start_time : -int(sample_rate / 6)]
    )  # weird pulse at the end of the model so remove last 1/6 second


# IPA Phonemizer: https://github.com/bootphon/phonemizer
