import torch


def split_shot_query(data, way, shot, query, ep_per_batch=1):
    img_shape = data.shape[1:]
    data = data.view(ep_per_batch, way, shot + query, *img_shape)
    x_shot, x_query = data.split([shot, query], dim=2)
    x_shot = x_shot.contiguous()
    x_query = x_query.contiguous().view(ep_per_batch, way * query, *img_shape)
    return x_shot, x_query


def make_nk_label(n, k, ep_per_batch=1):
    label = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
    label = label.repeat(ep_per_batch)
    return label

def split_shot_query_semi(data, way, shot, query, unlabel_num=0, ep_per_batch=1):
    img_shape = data.shape[1:]
    data = data.view(ep_per_batch, way, shot + query, *img_shape)
    x_shot, x_query = data.split([shot, query], dim=2)      # x_shot: b*N*K* c*h*w (3*84*84)
    x_shot_l, x_shot_ul = x_shot.split([shot-unlabel_num,unlabel_num],dim=2) 
    x_shot_l = x_shot_l.contiguous()
    x_shot_ul = x_shot_ul.contiguous()
    x_query = x_query.contiguous().view(ep_per_batch, way * query, *img_shape)
    return x_shot_l, x_shot_ul, x_query