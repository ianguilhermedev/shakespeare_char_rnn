import torch
import torch.nn.functional as F
from .text_processing import encode_text

def next_char(model,
              text,
              char_idx,
              idx_char,
              temperature=1):
    
    encoded = encode_text(text, char_idx).unsqueeze(dim=0)

    with torch.no_grad():
        logits = model(encoded)

        y_prob = F.softmax(logits[0, :, -1] / temperature, dim=-1)
        y_prob = torch.multinomial(y_prob, num_samples=1)

    return idx_char[y_prob.item()]

def append_text(model,
                text,
                char_idx,
                idx_char,
                num_chars=100,
                temperature=1):

    for _ in range(num_chars):
        text += next_char(model, text, char_idx, idx_char, temperature)
    return text