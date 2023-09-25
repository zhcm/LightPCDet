import torch
from torch.nn.utils import clip_grad_norm_


def train_one_epoch(model,
                    optimizer,
                    train_loader,
                    dataloader_iter,
                    lr_scheduler,
                    cur_epoch=None,
                    total_it_each_epoch=None,
                    accumulated_iter=0,
                    merge_all_iters_to_one_epoch=False,
                    tb_log=None,
                    use_amp=False):
    if not merge_all_iters_to_one_epoch:  # 不merge的话每个epoch都要初始化
        dataloader_iter = iter(train_loader)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0 ** 16))
    for cur_it in range(0, total_it_each_epoch):
        batch = next(dataloader_iter)
        lr_scheduler.step(accumulated_iter, cur_epoch)
        cur_lr = float(optimizer.lr)
        tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, tb_dict, disp_dict = model_func(model, batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        scaler.step(optimizer)
        scaler.update()

        accumulated_iter += 1  # 累计迭代次数

    return accumulated_iter
