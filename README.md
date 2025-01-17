# pix2tex - LaTeX OCR

This is a fork of [lukas-blecher/LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR) that retrains the model on the [normalized Im2latex-100k dataset](https://im2markup.yuntiandeng.com/data/) that has been put through [some additional preprocessing](https://github.com/Adi-UA/LaTeX-OCR/blob/main/scripts/download_and_extract_data.py#L64-L90).

The [details of my best run can be found on Weights & Biases](https://wandb.ai/adioss/LaTeX-OCR/runs/gzcrm3f6) and the [corresponding best model checkpoint can be downloaded from here](https://drive.google.com/drive/folders/1_i6vDSnAJT0d_j0uILBNlQgZPCrcUBze?usp=sharing).

## Testing

1. Create the virtual environment and install the required packages if you haven't already (or the training script would have done it for you):

   ```bash
   python3 -m venv venv && \
   source ./venv/bin/activate  && \
   pip3 install gpustat==1.1.1 wandb==0.15.12 && \
   pip3 install -e .[train]
   ```

2. Then, download and place the [best model checkpoint](https://drive.google.com/drive/folders/1_i6vDSnAJT0d_j0uILBNlQgZPCrcUBze?usp=sharing) in the `custom_checkpoints` directory, in a Python file do:

   ```python
   from munch import Munch
   from PIL import Image

   from pix2tex.cli import LatexOCR

   custom_config = custom_arguments = {
      "config": os.path.join(project_dir, "custom/train_tok_config.yaml"),
      "checkpoint": os.path.join(project_dir, "custom_checkpoints/model.pth"),
      "tokenizer": os.path.join(project_dir, "custom/train_tokenizer.json"),
      "no_cuda": True,
      "no_resize": False,
   }
   model = LatexOCR(Munch(custom_config))
   img = Image.open("path/to/image.png")
   prediction = model(img)
   print(prediction)
   ```

## Training

```
./scripts/train
```

The script will create a virtual environment, download and prepare the data, create the tokenizer and then start training. It will prompt you to enter the paths to the various files required for training. It has default values for the paths, so you can just press enter to use those too.

The default parameters are the same ones I used to train our best model with a BLEU score of 0.87. You can change them by editing the `train_tok_config.yaml` file in the `custom` directory.

## Results

The results of our best model on the test set have been saved to the `results` directory. It was created with the following commands:

1. Create the virtual environment and install the required packages if you haven't already (or the training script would have done it for you):

   ```bash
   python3 -m venv venv && \
   source ./venv/bin/activate  && \
   pip3 install gpustat==1.1.1 wandb==0.15.12 && \
   pip3 install -e .[train]
   ```

2. Download the [best model checkpoint](https://drive.google.com/drive/folders/1_i6vDSnAJT0d_j0uILBNlQgZPCrcUBze?usp=sharing) and put in in the `custom_checkpoints` directory.
3. Run the evaluation script:
   ```bash
   python3 scripts/get_im2latex100k_test_results.py
   ```

Everything else is the same as in the original repo.
