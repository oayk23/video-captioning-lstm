from transformers.models.x_clip import XCLIPModel,XCLIPProcessor
import torch
import av
import argparse

from src.config import Config
from src.model import LSTMDecoder
from src.utils import read_video_pyav,sample_frame_indices,greedy_decode
from src.constants import XCLIP_PATH
from src.tokenizer import Tokenizer



def inference(video_path,device,lstm_path,tokenizer_path):
    xclip_model = XCLIPModel.from_pretrained(XCLIP_PATH).to(device).eval()
    processor = XCLIPProcessor.from_pretrained(XCLIP_PATH)
    config = Config()
    lstm = LSTMDecoder(config).to(device).eval()
    lstm.load_state_dict(torch.load(lstm_path))
    tokenizer = Tokenizer(tokenizer_path)
    container = av.open(video_path)
    indices = sample_frame_indices(clip_len=16, frame_sample_rate=(container.streams.video[0].frames // 16), seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container, indices)
    inputs = processor(videos=list(video),return_tensors="pt").to(device) # type: ignore
    with torch.no_grad():
        video_features = xclip_model.get_video_features(**inputs) # type: ignore
        generated_caption = greedy_decode(model=lstm,video_features=video_features,bos_token_id=2,eos_token_id=3,max_len=50,device=device)
    text = tokenizer.decode(generated_caption)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")

    parser.add_argument("--video_path", type=str, help="Path of the video.")
    parser.add_argument("--device", type=str, default="cuda", help="Device,must be either cuda or cpu")
    parser.add_argument("--lstm_path", type=str, default="models/lstm.pt", help="trained model weights.")
    parser.add_argument("--tokenizer_path",type=str,default="tokenizer.model",help="trained sentencepiece tokenizer path.")

    args = parser.parse_args()
    generated_caption = inference(args.video_path,args.device,args.lstm_path,args.tokenizer_path)
    print(generated_caption)
