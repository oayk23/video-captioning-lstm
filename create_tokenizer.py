import sentencepiece as sp
import os
import json

from src.constants import VOCAB_SIZE

if __name__ == "__main__":

    annotation_path = os.path.join("MSRVTT","MSRVTT","annotation","MSR_VTT.json")
    with open(annotation_path,"r",encoding="utf-8") as annotation:
        annot = json.load(annotation)
    with open("all_captions.txt","w",encoding="utf-8") as writer:
        for data_point in annot['annotations']:
            writer.writelines(data_point['caption'] + "\n")
        writer.close()
    
    sp.SentencePieceTrainer.Train(f"--input=all_captions.txt --model_type=bpe --vocab_size={VOCAB_SIZE} --model_prefix=tokenizer --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --pad_piece=[PAD] --unk_piece=[UNK] --bos_piece=[BOS] --eos_piece=[EOS]")