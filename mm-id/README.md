# MM-ID

## benchmark

Question-answer pairs of the benchmark is at [benchmark_mmid.json](./benchmark/benchmark_mmid.json).

Images can be downloaded at [images](https://huggingface.co/jiyatai/IDA-VLM/blob/main/MM-ID/MMID_images.zip). Movie and matching images are collected from public datasets. Note that we download animation images from web and provide download urls of animation images at [here](./benchmark/images/animation/). You can use download.py to prepare images. If there are any copyright issues, please contact us, and we will remove the corresponding images.

For benchmark visualization, you can open [benchmark_mmid.md](./benchmark/benchmark_mmid.md) with a Markdown Reader.

## Eval

The evaluation code for closed-source APIs is [here](./eval/eval_api/).

Once you get prediction.json, you can use gpt4_eval.py to score the answers.
