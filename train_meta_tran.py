import argparse
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler
# from datasets.sampler2 import CategoriesSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

#torch.backends.cudnn.enabled = False

def adjust_learning_rate(optimizers, lr, iter):

    new_lr = lr * (0.1**(int(iter//5000)))
    #optimizers.param_groups[0]['lr'] = new_lr   
    #for optimizer in optimizers:
    for param_group in optimizers.param_groups:
        param_group['lr'] = new_lr

def make_adj_gt(label_tr, label, ep_per_batch):

    label_tr = label_tr.unsqueeze(1)
    label = label.unsqueeze(1)

    one_hot_s = torch.zeros((label_tr.size()[0],5), device=label_tr.device)
    one_hot_s.scatter_(1, label_tr, 1)
    one_hot_sf = one_hot_s.reshape(ep_per_batch, label_tr.size()[0]//ep_per_batch, 5)

    one_hot_q = torch.zeros((label.size()[0],5), device=label.device)
    one_hot_q.scatter_(1, label, 1)
    one_hot_qf = one_hot_q.reshape(ep_per_batch, label.size()[0]//ep_per_batch, 5)

    one_hot_t = torch.cat([one_hot_sf, one_hot_qf], dim=1)

    adj_gt = one_hot_t.bmm(one_hot_t.transpose(1,2))
    return adj_gt * 0.05


def main(config):
    svname = args.name
    if svname is None:
        svname = 'meta_{}-{}shot'.format(
                config['train_dataset'], config['n_shot'])
        svname += '_' + config['model'] + '-' + config['model_args']['encoder']
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
#    cp_exist = utils.ensure_path(save_path)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #### Dataset ####

    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']

    if config.get('n_train_way') is not None:
        n_train_way = config['n_train_way']
    else:
        n_train_way = n_way
    if config.get('n_train_shot') is not None:
        n_train_shot = config['n_train_shot']
    else:
        n_train_shot = n_shot
    if config.get('ep_per_batch') is not None:
        ep_per_batch = config['ep_per_batch']
    else:
        ep_per_batch = 1

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    utils.log('train dataset: {} (x{}), {}'.format(
            train_dataset[0][0].shape, len(train_dataset),
            train_dataset.n_classes))
    # utils.log('train dataset: {} (x{}), {}'.format(
    #     train_dataset[0][0].shape, len(train_dataset),
    #     train_dataset.num_class))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)
    train_sampler = CategoriesSampler(
            train_dataset.label, config['train_batches'],
            n_train_way, n_train_shot + n_query,
            ep_per_batch=ep_per_batch)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    # tval
    if config.get('tval_dataset'):
        tval_dataset = datasets.make(config['tval_dataset'],
                                     **config['tval_dataset_args'])
        utils.log('tval dataset: {} (x{}), {}'.format(
                tval_dataset[0][0].shape, len(tval_dataset),
                tval_dataset.n_classes))
        # utils.log('tval dataset: {} (x{}), {}'.format(
        #         tval_dataset[0][0].shape, len(tval_dataset),
        #         tval_dataset.num_class))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(tval_dataset, 'tval_dataset', writer)
        tval_sampler = CategoriesSampler(
               tval_dataset.label, 200,
               n_way, n_shot + n_query,
               ep_per_batch=4)
        # tval_sampler = CategoriesSampler(
        #         tval_dataset.label, config['test_batches'],
        #         n_way, n_shot + n_query,
        #         ep_per_batch=4)
        tval_loader = DataLoader(tval_dataset, batch_sampler=tval_sampler,
                                 num_workers=8, pin_memory=True)
    else:
        tval_loader = None

    # val
    val_dataset = datasets.make(config['val_dataset'],
                                **config['val_dataset_args'])
    utils.log('val dataset: {} (x{}), {}'.format(
            val_dataset[0][0].shape, len(val_dataset),
            val_dataset.n_classes))
    # utils.log('val dataset: {} (x{}), {}'.format(
    #         val_dataset[0][0].shape, len(val_dataset),
    #         val_dataset.num_class))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    val_sampler = CategoriesSampler(
            val_dataset.label, config['test_batches'],
            n_way, n_shot + n_query,
            ep_per_batch=4)
    #val_sampler = CategoriesSampler(
    #        val_dataset.label, 200,
    #        n_way, n_shot + n_query,
    #        ep_per_batch=4)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    ########

    #### Model and optimizer ####

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])

        if config.get('load_encoder'):
            encoder = models.load(torch.load(config['load_encoder'])).encoder
            model.encoder.load_state_dict(encoder.state_dict())

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])
    
    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    aves_keys = ['tl', 'ta', 'tvl', 'tva', 'vl', 'va'] #, 'tl2', 'tvl2', 'vl2']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []
    
    for epoch in range(1, max_epoch + 1):
        timer_epoch.s()
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        if config.get('freeze_bn'):
            utils.freeze_bn(model) 
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        np.random.seed(epoch)
        for data, _ in tqdm(train_loader, desc='train', leave=False):
            # print(data.size())
            x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_train_way, n_train_shot, n_query,
                    ep_per_batch=ep_per_batch)

            label_tr = fs.make_nk_label(n_train_way, n_train_shot,
                    ep_per_batch=ep_per_batch).cuda()
            label = fs.make_nk_label(n_train_way, n_query,
                    ep_per_batch=ep_per_batch).cuda()
            # print(label_tr.size())
            # print(label_tr)
            # adj_gt = make_adj_gt(label_tr, label, ep_per_batch)

            optimizer.zero_grad()
            
            # logits = model(x_shot, x_query)
            logits = model(x_shot, x_query, label_tr)
 
            logits = logits.reshape(-1, n_train_way)
            # print(logits.size())
            # print(n_train_way)
            # logits = logits.view(-1, n_train_way) # (batch*n_way*n_query) * n_way

            # adj_gt = adj_gt.unsqueeze(3).repeat(1,1,1,wl.size()[3])

            loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)
            # print(adj_gt.size())
            # print(wl.size())
            # loss2 = torch.sum(torch.norm(adj_gt-wl, dim=(1,2)))
            total_loss = loss #+ args.lamb*loss2

            total_loss.backward()
            #loss.backward()
            optimizer.step()
            
            # aves['tl2'].add(loss2.item())
            aves['tl'].add(total_loss.sum().item())
            aves['ta'].add(acc)

            logits = None; total_loss = None; loss = None; #loss2 = None
            
        # eval
        model.eval()

        for name, loader, name_l, name_a in [
                ('tval', tval_loader, 'tvl', 'tva'),
                ('val', val_loader, 'vl', 'va')]:
        # for name, loader, name_l, name_a, name_l2 in [
        #         ('tval', tval_loader, 'tvl', 'tva', 'tvl2'),
        #         ('val', val_loader, 'vl', 'va', 'vl2')]:

            if (config.get('tval_dataset') is None) and name == 'tval':
                continue
            np.random.seed(0)
            for data, _ in tqdm(loader, desc=name, leave=False):
                x_shot, x_query = fs.split_shot_query(
                        data.cuda(), n_way, n_shot, n_query,
                        ep_per_batch=4)
                label_tr = fs.make_nk_label(n_way, n_shot,
                        ep_per_batch=4).cuda()
                label = fs.make_nk_label(n_way, n_query,
                        ep_per_batch=4).cuda()

                # adj_gt = make_adj_gt(label_tr, label, ep_per_batch)
                
                with torch.no_grad():
                    # logits = model(x_shot, x_query)
                    logits = model(x_shot, x_query, label_tr)
                    logits = logits.reshape(-1, n_way)
                    # adj_gt = adj_gt.unsqueeze(3).repeat(1,1,1,wl.size()[3])
                    # logits = logits.view(-1, n_way)
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)
                    # loss2 = torch.sum(torch.norm(adj_gt - wl, dim=(1,2)))
                    total_loss = loss #+ args.lamb*loss2

                aves[name_l].add(total_loss.sum().item())
                # aves[name_l2].add(loss2.item())
                aves[name_a].add(acc)

        _sig = int(_[-1])

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
        utils.log('epoch {}, train {:.4f}|{:.4f}, tval {:.4f}|{:.4f}, '
        'val {:.4f}|{:.4f}, {} {}/{} (@{})'.format(
        epoch, aves['tl'], aves['ta'], aves['tvl'], aves['tva'],
        aves['vl'], aves['va'], t_epoch, t_used, t_estimate, _sig))
        # utils.log('epoch {}, train {:.4f},{:.4f}|{:.4f}, tval {:.4f},{:.4f}|{:.4f}, '
        #         'val {:.4f},{:.4f}|{:.4f}, {} {}/{} (@{})'.format(
        #         epoch, aves['tl'], aves['tl2'], aves['ta'], aves['tvl'], aves['tvl2'], aves['tva'],
        #         aves['vl'], aves['vl2'], aves['va'], t_epoch, t_used, t_estimate, _sig))

        writer.add_scalars('loss', {
            'train': aves['tl'],
            'tval': aves['tvl'],
            'val': aves['vl'],
            # 'train2': aves['tl2'],
            # 'tval2': aves['tvl2'],
            # 'val2': aves['vl2'],            
        }, epoch)
        writer.add_scalars('acc', {
            'train': aves['ta'],
            'tval': aves['tva'],
            'val': aves['va'],
        }, epoch)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),

            'training': training,
        }
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj,
                    os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))

        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train_meta_mini.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--lamb', type=float, default=0.01) #original 0.1
    parser.add_argument('--gpu', default='0')
    #parser.add_argument('--n_iter', default='2')
    #parser.add_argument('--ad', default='30')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)
