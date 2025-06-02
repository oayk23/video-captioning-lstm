import sentencepiece as spm

class Tokenizer:
    def __init__(self,tokenizer_path:str,max_length:int = 50):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        self.max_length = max_length

    def encode(self,text:str):
        return self.tokenizer.Encode(text,add_bos=True,add_eos=True)
    def pad(self,ids:list):
        length = len(ids)
        if length < self.max_length:
            padding_length = self.max_length - length
            padding_vec = padding_length * [self.tokenizer.pad_id()]
            ids += padding_vec
            return ids
        else:
            return ids[:50]
    def decode(self,ids:list):
        return self.tokenizer.Decode(ids)