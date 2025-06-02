import numpy as np
import torch
from torch.nn import functional as F

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = seg_len
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def greedy_decode(model, video_features, bos_token_id=2, eos_token_id=3, max_len=32, device='cuda'):

    model.eval()
    generated = [bos_token_id]

    with torch.no_grad():
        for _ in range(max_len):
            input_token = torch.tensor([generated], device=device)
            logits = model(video_features, input_token) 

            next_token = torch.argmax(F.log_softmax(logits[:, -1, :], dim=-1), dim=-1)
            token_id = next_token.item()

            if token_id == eos_token_id:
                break

            generated.append(token_id) # type: ignore

    return generated[1:]
