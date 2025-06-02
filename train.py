import tqdm
import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import random_split,DataLoader
from evaluate import load
import logging
import argparse
from transformers.models.x_clip import XCLIPModel,XCLIPProcessor
import os
import json
from matplotlib import pyplot as plt


from src.config import Config
from src.constants import VOCAB_SIZE,XCLIP_PATH,VIDEOS_PATH
from src.dataloader import MSRVTTDataset
from src.model import LSTMDecoder
from src.tokenizer import Tokenizer
from src.utils import read_video_pyav,sample_frame_indices,greedy_decode

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        filename="training.log",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w"
    )


def train(args):
    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    device = args.device

    data_path = os.path.join("MSRVTT","MSRVTT","annotation","MSR_VTT.json")
    with open(data_path,"r",encoding="utf-8") as file:
        data = json.load(file)
    
    logging.info("Data loaded.")
    xclip_processor = XCLIPProcessor.from_pretrained(XCLIP_PATH)
    xclip_model = XCLIPModel.from_pretrained(XCLIP_PATH).to(device).eval()
    logging.info("XCLIP has loaded.")
    tokenizer = Tokenizer("tokenizer.model")
    dataset = MSRVTTDataset(data=data,processor=xclip_processor,tokenizer=tokenizer,videos_path=VIDEOS_PATH) # type:ignore

    validation_percentage = 0.1
    test_percentage = 0.05
    train_percentage = 0.85
    train_dataset,val_dataset,test_dataset = random_split(dataset,lengths=[train_percentage,validation_percentage,test_percentage])
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset,batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset,batch_size=1)
    logging.info("Dataloaders has loaded.")
    config = Config()
    lstm = LSTMDecoder(config).to(device)
    optimizer = torch.optim.Adam(lstm.parameters(),lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    logging.info("Model and optimizer has loaded. Training starting...")
    train_losses = []
    val_losses = []
    for i in tqdm.tqdm(range(1,epochs+1)):
        lstm.train()
        logging.info(f"Epoch:{i} training starting...")
        for j,(pixel_values,x,y) in enumerate(tqdm.tqdm(train_dataloader)):
            pixel_values,x,y = pixel_values.to(device),x.to(device),y.to(device)
            with torch.no_grad():
                video_features = xclip_model.get_video_features(pixel_values)
            logits = lstm(video_features,x)
            loss = loss_fn(
                logits.view(-1,VOCAB_SIZE),
                y.view(-1)
            )
            optimizer.zero_grad()
            logging.info(f"Step {j},Loss:{loss.item():.4f}")
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        logging.info(f"Epoch:{i} validation starting...")
        lstm.eval()
        for k,(pixel_values,x,y) in enumerate(tqdm.tqdm(val_dataloader)):
            pixel_values,x,y = pixel_values.to(device),x.to(device),y.to(device)
            with torch.no_grad():
                video_features = xclip_model.get_video_features(pixel_values)
                logits = lstm(video_features,x)
            loss =loss_fn(
                logits.view(-1,VOCAB_SIZE),
                y.view(-1)
            )
            logging.info(f"Step {j} Val,Loss:{loss.item():.4f}")
            val_losses.append(loss.item())
    
    logging.info("Training completed. Visualizing metrics.")
    plt.plot(train_losses,label="Train")
    plt.plot(val_losses,label="Val")
    plt.legend()
    plt.savefig("losses.png")
    logging.info("losses has saved. Testing model on test loader.")
    def generate_caption_pixel_values(pixel_values):
        with torch.no_grad():
            video_features = xclip_model.get_video_features(pixel_values)
        generated_caption = greedy_decode(model=lstm,video_features=video_features,bos_token_id=2,eos_token_id=3,max_len=50,device="cuda")
        text = tokenizer.decode(generated_caption)
        return text
    generated_captions = []
    ground_truths = []
    for (pixel_values,x,y) in tqdm.tqdm(test_dataloader):
        pixel_values = pixel_values.to(device)
        generated_caption = generate_caption_pixel_values(pixel_values)
        generated_captions.append(generated_caption)
        ground_truths.append(tokenizer.decode(y[0].tolist()))
    logging.info("test completed. computing bleu score")
    bleu = load("bleu")
    bleu_score = bleu.compute(predictions=generated_captions,references=ground_truths)
    print(bleu_score)
    logging.info("bleu score computed.saving model state dict.")
    if not os.path.exists("models"):
        os.mkdir("models")
                          
    torch.save(lstm.state_dict(),"models/lstm.pt")
    logging.info("model saved.")


if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser(description="Model Training Script")

    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--device",type=str,default="cuda",help="device, cuda,cpu etc.")

    args = parser.parse_args()

    train(args)
