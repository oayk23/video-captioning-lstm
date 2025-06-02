# Video Captioning with Microsoft's XCLIP and LSTMs
---
This repo contains **Video Captioning** with XCLIP and LSTMs.

What is the purpose of this repo?

This repo's goal is understanding multimodality between vision(image,multi-image data) and natural language bindings.

### Which architectures used within this repo?

Main architectures is: A Video Vision Transformer(ViViT) and LSTMDecoder. The XCLIP takes video(or multi image) data and prepares *(Batch_size,512)* length arrays and we are calling this arrays **latents**.

Main training procedure is aligning this latent spaces with LSTMs memory, so LSTM could learn that "with this latent space i could generate captions like this.". So we need a vision backbone for this task and most suitable one was XCLIP.

### Which data acquired for this repo?

Data has been collected from [MSRVTT](https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip) dataset. Utilized the whole dataset for training.

The dataset consists 10.000 videos and 20 caption for each video,resulting 200.000 captions.For processing the dataset we used PyAv and NumPy libraries.

---
# Installation and Getting Started

This repository created in *Windows*. So for other platforms like **Linux** you need to download and extract dataset manually.

1. Clone the repository: `git clone https://github.com/oayk23/video-captioning-lstm`
2. Open a Terminal on project dir and type `pip install -r requirements.txt`. This will install the necessary python libraries.
3. Click the `download_dataset.bat` file. That will download and extract the dataset.
4. In the project terminal, type `python create_tokenizer.py`. That will generate `tokenizer.model` and `tokenizer.vocab`.
5. Then you need to train a model. For training you could use `python train.py`. This will generate **model.pt** within `models/` folder. You could manipulate the hyperparameters by adding arguments.For arguments you could check out `train.py`
6. For inference you need to specify the video_path correctly. You could also specify the other parameters like device(cpu or cuda),tokenizer_path,model_path etc.

That's all!

---
# Limitations and Challenges

Project has some limitations:
- Based on the 16 frames selected from video,**latent space vectors** highly varies. Thats why model sometimes couldn't learn from specific cases.
- Despite it's size and robustness, it's not as good as Vision Language Models like LLaVA-NeXT models. It can't process images like this models and it just describes video with a few words instead of 100-200 tokens.
- LSTMs not good as **Transformers** on the long-context text generation tasks. So maybe some of the data points includes relatively long-context. That's why maybe LSTM performance could be not good as Transformer models.
Don't look at the worst perspective, project also has some advantageous sides😊:
- Efficient deployment: Model is not fast as the object detection models but with low FPS it generates satisfaction results.
- It has low parameters, thats make it easy to run on edge-devices.

---
# Thank You for interested in this repo 😁.
Thank you for your time. I hope this repo helped you to understand multimodality. You could also look at my [Image Captioning With Pytorch](https://github.com/oayk23/image_captioning_pytorch) repo for Image Captioning.
 
