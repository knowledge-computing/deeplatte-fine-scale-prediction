import math

import torch

import numpy as np
from torch import nn
from scipy.optimize import curve_fit


def gaussian(h, r, s, n=0):
    return n + s * (1. - np.exp(- (h ** 2 / (r / 2.) ** 2)))


def get_fit_bounds(x, y):
    n = np.nanmin(y)
    r = np.nanmax(x)
    s = np.nanmax(y)
    return (0, [r, s, n])


def get_fit_func(x, y, model):
    try:
        bounds = get_fit_bounds(x, y)
        popt, _ = curve_fit(model, x, y, method='trf', p0=bounds[1], bounds=bounds)
        return popt
    except Exception as e:
        return [0, 0, 0]


def get_semivariogram(distances, variances, bins, thr):
    valid_variances, valid_bins = [], []
    for b in range(len(bins) - 1):
        left, right = bins[b], bins[b + 1]
        mask = (distances >= left) & (distances < right)
        if np.count_nonzero(mask) > thr:
            v = np.nanmean(variances[mask])
            d = np.nanmean(distances[mask])
            valid_variances.append(v)
            valid_bins.append(d)

    x, y = np.array(valid_bins), np.array(valid_variances)
    popt = get_fit_func(x, y, model=gaussian)
    return popt


class EmbeddingSVGLoss(nn.Module):

    def __init__(self, **kwargs):
        super(EmbeddingSVGLoss, self).__init__()

        self.device = kwargs.get('device', 'cpu')

    def forward(self, emb, y, out):  # x: [b x c x h x w], y/out: [b x 1 x h x w]

        kl_losses = 0.
        batch_size, _, num_rows, num_cols = emb.shape

        for b in range(batch_size):
            emb_flatten = torch.flatten(emb[b, ...], start_dim=1)
            y_flatten, out_flatten = torch.flatten(y[b, ...]), torch.flatten(out[b, ...])

            # get pairs of labeled locations
            y_mask = torch.nonzero(~torch.isnan(y_flatten)).view(-1)
            y_mask_pairs = torch.combinations(y_mask)
            p1, p2 = y_mask_pairs[:, 0], y_mask_pairs[:, 1]
            y_emb_sim = torch.mean((emb_flatten[:, p1] - emb_flatten[:, p2]) ** 2, dim=0)
            y_var = (y_flatten[p1] - y_flatten[p2]) ** 2 / 2

            # get pairs of unlabeled locations (random select due to computation limitation)
            out_mask = torch.randint(num_rows * num_cols, (y_mask.shape[0] * 10,)).to(self.device)
            out_mask_pairs = torch.combinations(out_mask)
            p1, p2 = out_mask_pairs[:, 0], out_mask_pairs[:, 1]
            out_emb_sim = torch.mean((emb_flatten[:, p1] - emb_flatten[:, p2]) ** 2, dim=0)
            out_var = (out_flatten[p1] - out_flatten[p2]) ** 2 / 2

            # rescale the distance
            def max_min_rescale(t, min_t, max_t):
                t -= min_t
                t /= (max_t - min_t)
                return t

            min_sim, max_sim = 0, torch.max(torch.cat([out_emb_sim, y_emb_sim])).detach()
            # min_sim = torch.min(torch.cat([out_emb_sim, y_emb_sim])).detach()
            y_emb_sim = max_min_rescale(y_emb_sim, min_sim, max_sim)
            out_emb_sim = max_min_rescale(out_emb_sim, min_sim, max_sim)

            # get range from labeled data
            bins = [i / 10 for i in range(11)]
            thr = y_mask_pairs.shape[0] / len(bins) * 0.01

            dis_arr = y_emb_sim.detach().cpu().data.numpy() ** 0.5
            var_arr = y_var.detach().cpu().data.numpy()
            popt = get_semivariogram(dis_arr, var_arr, bins, thr)
            r, s, n = popt

            kl_loss = 0.
            if 0.2 < r < 0.9 and 0 < n < s:
                valid_bins = [b for b in bins if b < r]

                for i in range(len(valid_bins) - 1):
                    left, right = valid_bins[i], valid_bins[i + 1]
                    mask1 = (y_emb_sim >= left ** 2) & (y_emb_sim < right ** 2)
                    mask2 = (out_emb_sim >= left ** 2) & (out_emb_sim < right ** 2)
                    if mask1.sum() > thr and mask2.sum() > thr:
                        mu1, var1 = torch.mean(y_var[mask1]), torch.std(y_var[mask1])
                        mu2, var2 = torch.mean(out_var[mask2]), torch.std(out_var[mask2])
                        if var1 > 0 and var2 > 0:
                            kl_loss += (torch.log(var2 ** 2 / var1 ** 2) - 1 +
                                        (var1 ** 2 + (mu1 - mu2) ** 2) / var2 ** 2) * 0.5

                #torch.isnan(kl_loss):
                #torch.log
                if kl_loss > 0.:
                    kl_loss = torch.log(kl_loss / len(valid_bins) if len(valid_bins) > 0 else 0.)
                    kl_losses += kl_loss

        return kl_losses
