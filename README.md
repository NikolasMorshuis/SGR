# Segmentation-Guided MRI Reconstruction

This is the codebase for the paper [Segmentation-guided MRI reconstruction for meaningfully diverse reconstructions](https://arxiv.org/pdf/2407.18026)

It is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion)

## Installation

Clone the repository:

```bash
git clone https://github.com/NikolasMorshuis/SGR
cd SGR
```

Install the dependencies:

via conda:
```bash
conda env create -f environment.yml
conda activate SGR
```

or create a venv via pip:
```bash
python -m venv sgr
source sgr/bin/activate
pip install -r requirements.txt
```

## Usage
Download the models from these links and put them into the models folder: 

[segmentation model](https://nc.mlcloud.uni-tuebingen.de/index.php/s/XSmbFWA2tQFxe2Q)

[diffusion model](https://nc.mlcloud.uni-tuebingen.de/index.php/s/dX6Y4M2wsRZadWg)

You need to download the skm-tea dataset, following the [official instructions](https://github.com/StanfordMIMI/skm-tea/blob/main/DATASET.md).
Note that the size of the complete dataset is 1.6TB.

Once downloaded, you need to set an environment variable that points to the dataset directory.:

```bash
export MEDDLR_DATASETS_DIR=/path/to/SKM-TEA
```

You can then start generating diverse samples using Segmentation-Guided Reconstruction (SGR) with the following command:

```bash
python scripts/image_sample.py --model_path models/model1000000.pt --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --batch_size 2 --num_samples 1 --timestep_respacing ddim100 --use_ddim True --clip_denoised False --acc 16.0 --debug False --div_const 0.005 --just_sample False --split test
```

for vanilla repeated sampling, just set the `--just_sample` flag to `True`:
```bash
python scripts/image_sample.py --model_path models/model1000000.pt --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --batch_size 2 --num_samples 1 --timestep_respacing ddim100 --use_ddim True --clip_denoised False --acc 16.0 --debug False --div_const 0.005 --just_sample True --split test
```

## Citation

If you find this code useful, please consider citing:
```bibtex
@misc{morshuis2024segmentationguidedmrireconstructionmeaningfully,
      title={Segmentation-guided MRI reconstruction for meaningfully diverse reconstructions}, 
      author={Jan Nikolas Morshuis and Matthias Hein and Christian F. Baumgartner},
      year={2024},
      eprint={2407.18026},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2407.18026}, 
}