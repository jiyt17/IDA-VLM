# Tuning Data Construction

We construct visual instruction tuning data with ID reference based on VCR, RefCOCO, Flickr30k and MovieNet.

We release the data processing code that transforms original data to instruction tuning data in this folder.

## The first-stage tuning data

The [instruction data](https://huggingface.co/jiyatai/IDA-VLM/blob/main/tuning/alpha_vcr_ref_f30k_llava.json) of the first-stage tuning is from VCR, RefCOCO, Flickr30k.

Users should prepare images of these datasets following:

## The second-stage tuning data

For MovieNet, we use GTP4v to annotate question-answer pairs and captions data.

The generated instruction tuning data can be downloaded from [here](https://huggingface.co/jiyatai/IDA-VLM/blob/main/tuning/beta_gpt4v_mix_mini_new.json).

To avoid downloading the whole MovieNet dataset for other researchers, we provide the involved images of MovieNet [here](https://huggingface.co/jiyatai/IDA-VLM/blob/main/tuning/imgs.zip).
