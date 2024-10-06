"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th

from guided_diffusion import logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import skm_tea as st

from skm.mri import LogLossPlus2, SenseModel
from skm.skm_utils import GetMRI

from utils.model_wrapper import UNet_Wrapper
import monai
import pandas as pd


def main():
    args = create_argparser().parse_args()

    assert args.split in ['train', 'val', 'test'], 'split should be either train, val or test'
    logger.configure()
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    model.cuda()
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    div_const = args.div_const
    seg_loss = LogLossPlus2()

    if args.just_sample:
        save_dir = os.path.join(args.save_base_dir, f'just_sample_{args.just_sample}', f'{args.split}', f'acc_{args.acc}',
                                'save_16')
    else:
        save_dir = os.path.join(args.save_base_dir, f'just_sample_{args.just_sample}', f'{args.split}', f'acc_{args.acc}',
                                f'div_{div_const}')
    if not os.path.exists(save_dir) and not args.debug:
        os.makedirs(save_dir)

    include_segmentation = True
    if include_segmentation:
        seg_model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=5,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        seg_model.load_state_dict(th.load(
            args.segmentation_model_path))
        seg_model.cuda()
        seg_model.eval()
        seg_model = UNet_Wrapper(seg_model, add_noise=True)
    else:
        seg_model = None

    MRIs = GetMRI(normalize=True, split=args.split, acc=args.acc, filter_slices=None)
    # Optional: filter for slices with segmentations:
    selection_df = pd.read_csv(f'utils/label_distribution_{args.split}.csv', index_col=0)
    # create column which is a sum of all classes:
    selection_df['sum_segs'] = selection_df['1'] + selection_df['2'] + selection_df['3'] + selection_df['4']
    selection_df = selection_df[selection_df['sum_segs'] > 0]
    selection = selection_df['label'].values
    selection = np.sort(selection)
    MRIs = th.utils.data.Subset(MRIs, selection)

    logger.log("sampling...")
    num_channels = 2
    for j in range(len(MRIs)):
        kspace, kspace_us, maps, mask, target, segmentation, norm_constant, scan_name = MRIs.__getitem__(j)
        kspace = kspace.cuda().unsqueeze(0)
        kspace_us = kspace_us.cuda().unsqueeze(0)
        maps = maps.cuda().unsqueeze(0)
        mask = mask.cuda().unsqueeze(0)
        target = target.cuda().unsqueeze(0)
        A = SenseModel(maps)

        sampled_image_array = []
        seg_array = []
        class_array = []

        if args.just_sample:
            specific_classes = th.tensor(np.arange(1))
        else:
            specific_classes = th.tensor(np.arange(5))[1:]

        for c, specific_class in enumerate(specific_classes):
            if args.just_sample:
                just_sample_here = True
                batch_size_here = 16
            else:
                just_sample_here = False
                batch_size_here = args.batch_size
            class_array += [specific_class]*batch_size_here
            specific_class = int(specific_class.item())

            model_kwargs = {}

            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                (batch_size_here, num_channels , 512, 160),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                kspace_us=kspace_us,
                mask=mask,
                seg_model=seg_model,
                seg_loss=seg_loss,
                div_const=div_const,
                specific_class=specific_class,
                just_sample=just_sample_here,
                A=A,
            )

            with th.no_grad():
                fully_sampled_seg = seg_model(target.abs())
                fully_sampled_seg = fully_sampled_seg.argmax(1)
                full_image = A(kspace, adjoint=True).abs()[0].cpu()
                orig_image = A(kspace_us, adjoint=True).abs()[0].cpu()
                sampled_image = th.complex(sample[:, 0], sample[:, 1]).abs()
                sampled_image = sampled_image.unsqueeze(1)
                sampled_image_array.append(sampled_image)

                output = seg_model(sampled_image)
                output_prob = th.nn.functional.softmax(output, dim=1)
                output_seg = output_prob.argmax(1)
                seg_array.append(output_seg)

        sampled_image_array = th.cat(sampled_image_array, dim=0)
        seg_array = th.cat(seg_array, dim=0)
        class_array = th.tensor(class_array)

        if not args.debug:
            th.save({'arr': sampled_image_array,
                     'perturbed_image': orig_image.unsqueeze(0).unsqueeze(0).float(),
                     'x_orig': full_image.unsqueeze(0).unsqueeze(0).float(),
                     'segmentations': seg_array.cpu().unsqueeze(1).type(th.uint8),
                     'fully_sampled_seg': fully_sampled_seg[0].cpu().unsqueeze(0).unsqueeze(0).type(th.uint8),
                     'class_array': class_array,
                     'scan_name': scan_name,
                     'norm_constant': norm_constant}, f'{save_dir}/{scan_name}_{j:04d}.pth')
            logger.log(f'Saved {scan_name}_{j:04d}.pth')

    logger.log("done")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=4,
        num_channels=2,
        use_ddim=True,
        model_path="models/model1000000.pt",
        segmentation_model_path="models/model_segmentation.pt"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    acceleration = {'acc': 16.0}
    add_dict_to_argparser(parser, acceleration)
    split = {'split': 'test'}
    add_dict_to_argparser(parser, split)
    plot = {'plot': False}
    add_dict_to_argparser(parser, plot)
    just_sample = {'just_sample': False}
    add_dict_to_argparser(parser, just_sample)
    debug = {'debug': False}
    add_dict_to_argparser(parser, debug)
    div_const = {'div_const': 1000.0}
    add_dict_to_argparser(parser, div_const)
    save_base_dir = {'save_base_dir': './results/skm'}
    add_dict_to_argparser(parser, save_base_dir)
    return parser


if __name__ == "__main__":
    main()
