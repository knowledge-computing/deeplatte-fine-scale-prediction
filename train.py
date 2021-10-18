import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dat
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter

from models.deeplatte import DeepLatte
from models.st_losses import OneStepSpatialLoss
from models.svg_losses import EmbeddingSVGLoss
from options.train_options import parse_args
from scripts.data_loader import load_data_from_file
from scripts.early_stopping import EarlyStopping


def train():
    """
    b: batch_size
    c: num_channels / num_features
    s: seq_len
    h: num_rows
    w: num_cols
    o: output_dim
    """

    """ construct index-based data loader """
    idx = np.array([i for i in range(args.seq_len + 1, data_obj.num_times)])
    idx_dat = dat.TensorDataset(torch.tensor(idx, dtype=torch.int32))
    train_idx_data_loader = dat.DataLoader(dataset=idx_dat, batch_size=args.batch_size, shuffle=True)

    idx = np.array([i for i in range(args.seq_len + 1, data_obj.num_times)])
    idx_dat = dat.TensorDataset(torch.tensor(idx, dtype=torch.int32))
    test_idx_data_loader = dat.DataLoader(dataset=idx_dat, batch_size=1, shuffle=False)

    """ set writer, loss function, and optimizer """
    mse_loss_func = nn.MSELoss()
    mse_sum_loss_func = nn.MSELoss(reduction='sum')
    spatial_loss_func = OneStepSpatialLoss()
    svg_loss_func = EmbeddingSVGLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=args.patience, verbose=args.verbose)

    def construct_sequence_x(idx_list, dynamic_x, static_x):
        d_x = [dynamic_x[i - args.seq_len + 1: i + 1, ...] for i in idx_list]  # b x [s x c x h x w]
        d_x = np.stack(d_x, axis=0)  # [b x s x c x h x w]
        s_x = np.expand_dims(static_x, axis=0)  # [1 x 1 x c x h x w]
        s_x = np.repeat(s_x, args.seq_len, axis=1)  # [1 x s x c x h x w]
        s_x = np.repeat(s_x, len(idx_list), axis=0)  # [b x s x c x h x w]
        x = np.concatenate([d_x, s_x], axis=2)  # [b x s x c x h x w]
        return torch.tensor(x, dtype=torch.float).to(device)

    def construct_y(idx_list, output_y):
        y = [output_y[i] for i in idx_list]
        y = np.stack(y, axis=0)  # [b x o x h x w] where o = 1
        return torch.tensor(y, dtype=torch.float).to(device)

    """ training """
    for epoch in range(args.num_epochs):

        model.train()
        total_losses, train_losses, val_losses, l1_losses, ae_losses, sp_losses, tp_losses = 0, 0, 0, 0, 0, 0, 0

        for _, idx in enumerate(train_idx_data_loader):
            batch_idx = idx[0]

            """ construct sequence input """
            batch_x = construct_sequence_x(batch_idx, data_obj.dynamic_x, data_obj.static_x)  # [b x t x c x h x w]
            batch_y = construct_y(batch_idx, data_obj.train_y)  # [b x 1 x h x w]
            batch_val_y = construct_y(batch_idx, data_obj.val_y)

            """ start train """
            model(batch_x)
            out, sparse_x, en_x, de_x, emb = model(batch_x)
            train_loss = mse_loss_func(batch_y[~torch.isnan(batch_y)], out[~torch.isnan(batch_y)])
            train_losses += train_loss.item()

            total_loss = train_loss

            # """ add loss according to the model type """
            if 'l1' in model_types:
                l1_loss = model.sparse_layer.l1_loss()
                l1_losses += l1_loss.item()
                total_loss += l1_loss * args.alpha

            # if 'ae' in model_types:
            #     ae_loss = mse_sum_loss_func(sparse_x, de_x)
            #     ae_losses += ae_loss.item()
            #     total_loss += ae_loss * args.beta

            if 'st' in model_types:
                sp_loss = spatial_loss_func(out)  # or using embeddings from convlstm
                sp_losses += sp_loss.item()

                pre_batch_idx = batch_idx - torch.ones_like(batch_idx)
                pre_batch_x = construct_sequence_x(pre_batch_idx, data_obj.dynamic_x,
                                                   data_obj.static_x)  # [b x s x c x h x w]
                pre_out, _, _, _, _ = model(pre_batch_x)
                tp_loss = mse_loss_func(out, pre_out)  # or using embeddings from convlstm
                tp_losses += tp_loss.item()

                total_loss += (sp_loss + tp_loss) * args.gamma

            if 'svg' in args.model_types:
                svg_loss = svg_loss_func(emb, batch_y, out)
                total_loss += svg_loss * args.eta

            total_losses += total_loss.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            """ validate """
            val_loss = mse_loss_func(batch_val_y[~torch.isnan(batch_val_y)], out[~torch.isnan(batch_val_y)])
            val_losses += val_loss.item()

        if args.verbose:
            logging.info('Epoch [{}/{}] total_loss = {:.3f}, train_loss = {:.3f}, val_loss = {:.3f}, '
                         'l1_losses = {:.3f}, ae_losses = {:.3f}, sp_losses = {:.3f}.'
                         .format(epoch, args.num_epochs, total_losses, train_losses, val_losses,
                                 l1_losses, ae_losses, sp_losses))

        # write for tensor board visualization
        if args.use_tb:
            tb_writer.add_scalar('data/train_loss', train_losses, epoch)
            tb_writer.add_scalar('data/val_loss', val_losses, epoch)

        # early_stopping
        early_stopping(val_losses, model, model_file)

        # evaluate testing data
        if len(data_obj.test_loc) > 0:

            model.eval()

            pred = []
            with torch.no_grad():
                for i, data in enumerate(test_idx_data_loader):
                    batch_idx = data[0]
                    batch_x = construct_sequence_x(batch_idx, data_obj.dynamic_x, data_obj.static_x)
                    out, _, _, _, _ = model(batch_x)
                    pred.append(out.cpu().data.numpy())

            pred = np.concatenate(pred)
            test_y = data_obj.test_y[args.seq_len + 1:, ...]

            rmse = np.nanmean((test_y - pred) ** 2) ** 0.5
            mape = np.nanmean(np.abs((test_y - pred) / test_y)) * 100
            r2 = r2_score(test_y[~np.isnan(test_y)], pred[~np.isnan(test_y)])

            if args.verbose:
                logging.info('Epoch [{}/{}] testing: rmse = {:.3f}, mape = {:.3f}, r2 = {:.3f}.'
                             .format(epoch, args.num_epochs, rmse, mape, r2))

        if early_stopping.early_stop:
            break


if __name__ == '__main__':

    args = parse_args()
    #device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')  # the gpu device
    device = torch.device('cpu')

    """ tensor board """
    if args.use_tb:
        tb_writer = SummaryWriter(args.tb_path)
    else:
        tb_writer = None

    """ load data """
    if os.path.exists(args.data_path):
        data_obj = load_data_from_file(args.data_path)
    else:
        raise NotImplementedError

    """ load model """
    model_types = args.model_types.split(',')
    model = DeepLatte(in_features=data_obj.num_features,
                      en_features=[int(i) for i in args.en_features.split(',')],
                      de_features=[int(i) for i in args.de_features.split(',')],
                      in_size=(data_obj.num_rows, data_obj.num_cols),
                      h_channels=args.h_channels,
                      kernel_sizes=[int(i) for i in args.kernel_sizes.split(',')],
                      num_layers=1,
                      fc_h_features=args.fc_h_features,
                      out_features=1,  # fixed
                      p=0.5,
                      device=device).to(device)

    model_file = os.path.join(args.model_dir, args.model_name + '.pkl')
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))

    train()

