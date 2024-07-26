import torch
import numpy as np

class UNet_Wrapper(torch.nn.Module):
    """
    This class should do test-time augmentation. It should mirror input image x in all 4 directions.
    """
    def __init__(self, model, add_noise=False):
        super().__init__()
        self.model = model
        self.add_noise = add_noise

    def forward(self, x):
        len_x = x.shape[0]
        # mirror the image in all 4 directions:
        x = torch.stack([x, x.flip(-2), x.flip(-1), x.flip(-2,-1)], dim=0)
        x = x.unsqueeze(1) # dimension for repeats
        if self.add_noise is not False:
            # create 4 different noise levels with sd 0, 0.2, 0.4, 0.6:
            repeats = 1
            x = x.repeat(1, repeats, 1, 1, 1, 1)
            #sd = torch.tensor(np.repeat([0.1, 0.2], repeats*len_x), device=x.device, dtype=x.dtype)
            sd = torch.tensor([[0.05]], device=x.device, dtype=x.dtype)  # or [[0.1, 0.5]]?
            sd = sd.repeat((x.shape[0], 1))
            noise = torch.randn_like(x) * sd.view(4, repeats, 1, 1, 1, 1)
            x = x + noise
            x = torch.clamp(x, 0)
            same_indices = [np.arange(0, len_x) + 4*i*len_x for i in range(repeats)]
            same_indices = np.array(same_indices).flatten()
            #same_indices = np.array([0, 1, 8, 9])
        else:
            repeats = 1
            same_indices = np.arange(0, len_x)
        # forward pass:
        x = x.view(-1, *x.shape[3:])
        output = self.model(x)
        output = output.view(4, repeats, len_x, 5, *x.shape[2:])
        # do the opposite mirroring to get the original image back:


        output = torch.stack([output[0], output[1].flip(-2), output[2].flip(-1), output[3].flip(-2,-1)], dim=0)
        # average the output:
        output = output.mean(dim=0)
        output = output.mean(dim=0)
        return output