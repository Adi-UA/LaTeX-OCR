# Description: Script to train pix2tex model

echo Install pix2tex training dependencies
python3 -m venv .venv
source .venv/bin/activate
pip3 install pix2tex[train]
pip3 install gpustat
pip3 install opencv-python-headless
pip install wandb

sleep 1
echo Check GPU status
gpustat

sleep 1
echo Prepare dataset and removing old version
python3 adi_prepare_data.py

sleep 1
echo Generate Lukas pickle files
python3 generate_lukas_data_split.py
python3 -m pix2tex.dataset.dataset -i dataset/data/train -e dataset/data/train.lst -o dataset/data/train.pkl
python3 -m pix2tex.dataset.dataset -i dataset/data/val -e dataset/data/val.lst -o dataset/data/val.pkl
python -m pix2tex.dataset.dataset --equations dataset/data/im2latex_formulas.final.lst --vocab-size 8000 --out tokenizer.json

echo Generating config file
echo {gpu_devices: [2], backbone_layers: [2, 3, 7], betas: [0.9, 0.999], batchsize: 10, bos_token: 1, channels: 1, data: dataset/data/train.pkl, debug: true, decoder_args: {'attn_on_attn': true, 'cross_attend': true, 'ff_glu': true, 'rel_pos_bias': false, 'use_scalenorm': false}, dim: 256, encoder_depth: 4, eos_token: 2, epochs: 50, gamma: 0.9995, heads: 8, id: null, load_chkpt: null, lr: 0.001, lr_step: 30, max_height: 192, max_seq_len: 512, max_width: 672, min_height: 32, min_width: 32, model_path: checkpoints, name: mixed, num_layers: 4, num_tokens: 8000, optimizer: Adam, output_path: outputs, pad: false, pad_token: 0, patch_size: 16, sample_freq: 2000, save_freq: 1, scheduler: StepLR, seed: 42, temperature: 0.2, test_samples: 5, testbatchsize: 20, tokenizer: tokenizer.json, valbatches: 100, valdata: dataset/data/val.pkl} > custom.yaml

echo Start training
python3 -m pix2tex.train --config custom.yaml
