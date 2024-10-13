# Model

<img src="../fig/idavlm.png">

## Base

IDA-VLM is fine-tuned from [Qwen-VL-Chat](https://github.com/QwenLM/Qwen-VL). We adopt a dual-stage fine-tuning method to train the model on identity memory and recognition across diverse scenes.

The initial phase leverages annotations in VCR, Flickr30k and RefCOCO, while the second stage tuning data is based on MovieNet. We mix tuning data from LLaVA and ShareGPT4V into our data at a ratio of around 10%.

The running environment is the same as Qwen-VL's. Our finetuning code:

> bash finetune/finetune_ds.sh

To inference on MM-ID:

> python inference_test.py

In the newest version, we use vcr(30k), f30k(30k), RefCOCO(20k), llava(10k) in the first stage tuning, and MovieNet(60k), llava(6k), sharegpt4v(4k) in the second stage. Under this condition, IDA-VLM has comprehensive highest performance. The fine-tuned model weight can be downloaded from [here](https://huggingface.co/jiyatai/IDA-VLM/tree/main/weights/model-base).

More model details can be found at [Qwen-VL-Chat](https://github.com/QwenLM/Qwen-VL).

You can also use our constructed tuning data to unleash the potential of other LVLMs in ID recognition. Note that the required base model should be able to handle multiple images.

## ID-Former

We also introduce a specialized component, termed ID-Former, to enhance the model’s capability of recognizing character identities.

<details>
  <summary>To utilize ID-Former</summary>

ID-Former's architecture can be seen in Line 153 of visual.py in model folder. After you download the model weight, the same files can be found in weight folder.

Note that if you want to train with ID-Former, you need open 'use_llava', which will use two dataloader, one is for ID reference tuning data and the other one is for llava, sharegpt4v training data.

To load two dataloaders, you need replace trainer.py in your transformers lib.

</details>

The weights of IDA-VLM with ID-Former is [here](https://huggingface.co/jiyatai/IDA-VLM/tree/main/weights/model-idf).

## Easy Start

This is an example for using IDA-VLM, and more examples can be found in MM-ID. You can perpare your own images for test according to our data format.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

model_path = "/path/to/IDA-VLM/weights"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

question = "Heizi is <img>./example/Heizi_1.jpg</img>\n Leizi is <img>./example/Leizi_2.jpg</img>\n Chuchun is <img>./example/Chuchun_1.jpg</img>\n In the image: <img>./example/Chaopao_23</img>\n What is Chuchun doing?"
response, history = model.chat(tokenizer, query=question, history=None)
print('user:', question)
print('assistant: ', response)
```



