import torch

def encode_text(text, char_idx):
    encoded_text = [char_idx[char] for char in text.lower()]
    return torch.tensor(encoded_text)

def decode_text(idxs, idx_char):
    decoded_text = [idx_char[idx.item()] for idx in idxs]
    return "".join(decoded_text)

