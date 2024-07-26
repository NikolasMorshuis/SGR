import os

#os.environ["MEDDLR_DATASETS_DIR"] = "/mnt/qb/baumgartner/rawdata/SKM-TEA"

from pprint import pprint
import os

import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt

import dosma as dm

import meddlr.ops as oF
from meddlr.data import DatasetCatalog, MetadataCatalog
from meddlr.utils.logger import setup_logger
from meddlr.utils import env

from skm.mri import SenseModel
import unittest

# Set the default device if cuda is enabled
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Device: ", DEVICE)

# Run this setup phase only once.
# Otherwise, you may get multiple print statements
setup_logger()
logger = setup_logger("skm_tea")
path_mgr = env.get_path_manager()

# Some general utilities

from typing import Union, Sequence


def get_scaled_image(
        x: Union[torch.Tensor, np.ndarray], percentile=0.99, clip=False
):
    """Scales image by intensity percentile (and optionally clips to [0, 1]).

    Args:
      x (torch.Tensor | np.ndarray): The image to process.
      percentile (float): The percentile of magnitude to scale by.
      clip (bool): If True, clip values between [0, 1]

    Returns:
      torch.Tensor | np.ndarray: The scaled image.
    """
    is_numpy = isinstance(x, np.ndarray)
    if is_numpy:
        x = torch.as_tensor(x)

    scale_factor = torch.quantile(x, percentile)
    x = x / scale_factor
    if clip:
        x = torch.clip(x, 0, 1)

    if is_numpy:
        x = x.numpy()

    return x


def plot_images(
        images, processor=None, disable_ticks=True, titles: Sequence[str] = None,
        ylabel: str = None, xlabels: Sequence[str] = None, cmap: str = "gray",
        show_cbar: bool = False, overlay=None, opacity: float = 0.3,
        hsize=5, wsize=5, axs=None
):
    """Plot multiple images in a single row.

    Add an overlay with the `overlay=` argument.
    Add a colorbar with `show_cbar=True`.
    """

    def get_default_values(x, default=""):
        if x is None:
            return [default] * len(images)
        return x

    titles = get_default_values(titles)
    ylabels = get_default_values(images)
    xlabels = get_default_values(xlabels)

    N = len(images)
    if axs is None:
        fig, axs = plt.subplots(1, N, figsize=(wsize * N, hsize))
    else:
        assert len(axs) >= N
        fig = axs.flatten()[0].get_figure()

    for ax, img, title, xlabel in zip(axs, images, titles, xlabels):
        if processor is not None:
            img = processor(img)
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel(xlabel)

    if overlay is not None:
        for ax in axs.flatten():
            im = ax.imshow(overlay, alpha=opacity)

    if show_cbar:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

    if disable_ticks:
        for ax in axs.flatten():
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    return axs


# MRI dataset:
class GetMRI(torch.utils.data.Dataset):
    def __init__(self, split='train', acc=6.0, echo=1, normalize=True, seg_dir='gt', filter_slices=[100, 400]):
        """
        Echo is not yet implemented, we always focus on echo1
        normalize: divide by 99th percentile
        """
        dataset_dicts_train = DatasetCatalog.get("skmtea_v1_train")
        dataset_dicts_val = DatasetCatalog.get("skmtea_v1_val")
        dataset_dicts_test = DatasetCatalog.get("skmtea_v1_test")
        self.split = split
        self.acc = acc
        self.echo = echo
        self.normalize = normalize
        self.seg_dir = seg_dir
        self.filter_slices = filter_slices
        if self.filter_slices is None:
            self.slices_per_patient = 512
            self.filter_slices = [0, 512]
        else:
            self.slices_per_patient = filter_slices[1] - filter_slices[0]
        if split == 'train':
            self.dataset_dicts = dataset_dicts_train
        elif split == 'val':
            self.dataset_dicts = dataset_dicts_val
        elif split == 'test':
            self.dataset_dicts = dataset_dicts_test

    def __len__(self):
        return len(self.dataset_dicts) * self.slices_per_patient

    def __getitem__(self, idx):
        scan = self.dataset_dicts[idx//self.slices_per_patient]
        # Reconstruction data
        sl = idx % self.slices_per_patient + self.filter_slices[0]
        mask_str = 'masks/poisson_{}x'.format(self.acc)
        recon_file = scan["recon_file"]
        with h5py.File(recon_file, "r") as f:
            kspace = f["kspace"][sl, :, :, 0, :]  # we just load echo 1
            target = f["target"][sl, :, :, 0, 0]  # Shape: (x, ky, kz, #echos, #maps) - #maps = 1 for SKM-TEA
            maps = f["maps"][sl, :, :, :, :]  # Shape: (x, ky, kz, #coils, #maps) - maps are the same for both echos
            mask = torch.as_tensor(f[mask_str][()]).unsqueeze(0)

        if self.seg_dir == 'gt':
            seg_file = scan["gw_corr_mask_file"]
            segmentation = dm.read(seg_file).A[sl, ...]
            segmentation = torch.from_numpy(segmentation).unsqueeze(0)
            """ 
            labels:
            1. Patellar Cartilage
            2. Femoral Cartilage
            3. Tibial Cartilage - Medial
            4. Tibial Cartilage - Lateral
            5. Meniscus - Medial
            6. Meniscus - Lateral
            """
            segmentation = torch.where(segmentation == 4, 3, segmentation)
            segmentation = torch.where(segmentation == 5, 4, segmentation)
            segmentation = torch.where(segmentation == 6, 4, segmentation)
        else:
            file_name = scan['file_name'].split('.')[0] + '_' + '{:03}'.format(sl) + '.npy'
            segmentation = torch.from_numpy(np.load(os.path.join(self.seg_dir, file_name))).unsqueeze(0)

        mask = oF.zero_pad(mask, kspace.shape[0:2])
        # get image form kspace

        maps = torch.from_numpy(maps)[:, :, :, 0].unsqueeze(0).type(torch.complex128)
        mask = mask.unsqueeze(-1)
        kspace = torch.from_numpy(kspace).type(torch.complex128)
        kspace = kspace.unsqueeze(0)
        target = torch.from_numpy(target).unsqueeze(0)

        kspace_us = kspace * mask

        if self.normalize:
            A = SenseModel(maps)
            norm_constant = A(kspace_us, adjoint=True).abs()
            # get 99th percentile:
            norm_constant = torch.quantile(norm_constant, 0.99)
            kspace_us = kspace_us / norm_constant
            kspace = kspace / norm_constant
            target = target / norm_constant
        else:
            norm_constant = 1.0

        kspace = kspace.squeeze(0)
        kspace_us = kspace_us.squeeze(0)
        maps = maps.squeeze(0)
        mask = mask.squeeze(0)
        scan_name = scan['file_name'].split('.')[0] + '_' + '{:03}'.format(sl)

        return kspace, kspace_us, maps, mask, target, segmentation, norm_constant, scan_name


class TestGetMRI(unittest.TestCase):
    def test_get_mri(self):
        dataset = GetMRI(split='test', acc=6.0, echo=1, normalize=True, seg_dir='/mnt/qb/work/baumgartner/jmorshuis45/projects/guided-diffusion/predictions/seg_test_ddim100_acc8.0_fulldcTrue')
        item = dataset[0]

    def test_get_mri_no_seg(self):
        dataset = GetMRI(split='train', acc=6.0, echo=1, normalize=True,
                         seg_dir='/mnt/qb/work/baumgartner/jmorshuis45/projects/guided-diffusion/predictions/seg_test_ddim100_acc8.0_fulldcTrue')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=8)
        for i, data in enumerate(dataloader):
            kspace, kspace_us, maps, mask, target, segmentation, norm_constant, scan_name = data
            print(kspace.shape, kspace_us.shape, maps.shape, mask.shape, target.shape, segmentation, norm_constant,
                  scan_name)
            self.assertEqual(kspace.shape, torch.Size([1, 512, 160, 2]))
            self.assertEqual(kspace_us.shape, torch.Size([1, 512, 160, 2]))
            self.assertEqual(maps.shape, torch.Size([1, 512, 160, 2]))
            self.assertEqual(mask.shape, torch.Size([1, 512, 160, 1]))
            self.assertEqual(target.shape, torch.Size([1, 512, 160, 1]))


if __name__ == "__main__":
    # Load list of dictionaries for the SKM-TEA v1 training dataset.
    dataset_dicts_train = DatasetCatalog.get("skmtea_v1_train")
    dataset_dicts_val = DatasetCatalog.get("skmtea_v1_val")
    dataset_dicts_test = DatasetCatalog.get("skmtea_v1_test")
    split = 'test'
    output_dir = '/mnt/qb/baumgartner/rawdata/SKM-TEA/skm-tea/fully_sampled_echo_1/{}'.format(split)
    output_dir_label = '/mnt/qb/baumgartner/rawdata/SKM-TEA/skm-tea/fully_sampled_echo_1_label/{}'.format(split)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir_label):
        os.makedirs(output_dir_label)
    if split == 'train':
        dataset_dicts = dataset_dicts_train
    elif split == 'val':
        dataset_dicts = dataset_dicts_val
    elif split == 'test':
        dataset_dicts = dataset_dicts_test
    display = False
    for scan in dataset_dicts:
        print(scan["recon_file"])
        pprint(scan)
        for sl in range(scan['matrix_shape'][0]):
            # Reconstruction data
            recon_file = scan["recon_file"]
            with h5py.File(recon_file, "r") as f:
                kspace = f["kspace"][sl, :, :, :, :]  # Shape: (x, ky, kz, #echos, #coils)
                target = f["target"][sl, :, :, 0, 0]  # Shape: (x, ky, kz, #echos, #maps) - #maps = 1 for SKM-TEA
                maps = f["maps"][sl, :, :, :, :]  # Shape: (x, ky, kz, #coils, #maps) - maps are the same for both echos
                mask = torch.as_tensor(f["masks/poisson_6.0x"][()]).unsqueeze(0)

            mask = oF.zero_pad(mask, kspace.shape[0:2])
            # get image form kspace

            maps = torch.from_numpy(maps).to(DEVICE)[:, :, :, 0].unsqueeze(0).type(torch.complex128)
            mask = mask.to(DEVICE).unsqueeze(-1)
            A = SenseModel(maps)
            kspace = torch.from_numpy(kspace).to(DEVICE).type(torch.complex128)
            kspace = kspace[:, :, 0].unsqueeze(0)

            undersampled_kspace = kspace * mask
            # forward:
            # back to image
            # img_us = A(undersampled_kspace, adjoint=True)
            # back to kspace
            # kspace_us = A(img_us, adjoint=False)

            image = A(kspace, adjoint=True)

            image_bu = image.clone()

            # back to kspace
            kspace2 = A(image, adjoint=False)
            # back to image
            image = A(kspace2, adjoint=True)

            is_it_close = torch.isclose(image.abs(), image_bu.abs()).sum()
            # Yes it is!
            target = torch.from_numpy(target).to(DEVICE)

            # try conjugate gradient:
            x = image  # torch.zeros_like(image)
            A = SenseModel(maps)
            b = A(undersampled_kspace, adjoint=True)
            x = A.CG(undersampled_kspace, x, mask=mask, max_iter=3)

            # Segmentation data
            seg_file = scan["gw_corr_mask_file"]
            segmentation = dm.read(seg_file).A[sl, ...]  # Shape: (x, y, z)

            # Plot reconstructed image
            mag_img = np.abs(image)

            # save mag_img and segmentation
            output_name = os.path.join(output_dir, scan['file_name'].split('.')[0] + '_' + '{:03}'.format(sl) + '.npy')
            output_name_label = os.path.join(output_dir_label,
                                             scan['file_name'].split('.')[0] + '_' + '{:03}'.format(sl) + '_label.npy')
            # np.save(output_name, mag_img)
            # np.save(output_name_label, segmentation.astype(np.uint8))

    if display:
        _ = plot_images(
            [mag_img[..., 0, 0], mag_img[..., 1, 0]],  # echo1, echo2
            processor=lambda x: get_scaled_image(x, 0.95, clip=True),
            titles=["Echo 1", "Echo 2"],
            overlay=seg_colorized,
            opacity=0.4,
            hsize=5, wsize=2.3
        )

        print('ok')
