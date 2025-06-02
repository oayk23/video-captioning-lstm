from torch.utils.data import Dataset
from transformers.models.x_clip import XCLIPProcessor
import os
import av
import torch

from .tokenizer import Tokenizer
from .utils import sample_frame_indices,read_video_pyav

class MSRVTTDataset(Dataset):
    def __init__(self,
                 processor,
                 data:dict,
                 tokenizer:Tokenizer,
                 videos_path:str
                 ):
        self.processor = processor
        self.data = data
        self.videos_path = videos_path
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data['annotations'])

    def __getitem__(self,idx):
        data_dict = self.data['annotations'][idx]
        caption = data_dict['caption']
        video_id = data_dict['image_id']+".mp4"
        container = av.open(file=os.path.join(self.videos_path,video_id))
        indices = sample_frame_indices(clip_len=16, frame_sample_rate=(container.streams.video[0].frames // 16), seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container, indices)
        pixel_values = self.processor(videos=list(video),return_tensors="pt")['pixel_values'].squeeze(0) # type:ignore
        caption_encoded = self.tokenizer.encode(caption)
        x = self.tokenizer.pad(caption_encoded[:-1])
        y = self.tokenizer.pad(caption_encoded[1:])
    

        return pixel_values,torch.tensor(x,dtype=torch.int64),torch.tensor(y,dtype=torch.int64)
