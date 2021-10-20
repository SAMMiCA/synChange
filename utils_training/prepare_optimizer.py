import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim

def prepare_optim(args,model):
    # Optimizer
    if args.optim == 'adamw': # GLU-CHANGENet
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=args.lr,
                       weight_decay=args.weight_decay)
    elif args.optim =='adam': # DR-TANet

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, betas=(0.9, 0.999),
                               weight_decay=args.weight_decay)
    else: raise NotImplementedError

    # Scheduler
    if args.scheduler == 'multstep': # GLU-CHANGENet
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=args.milestones,
                                             gamma=0.5)
    elif args.scheduler == 'lambda': # DR-TANet
        lambda_lr = lambda epoch: (float)(args.n_epoch - epoch) / (float)(args.n_epoch)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    else: raise NotImplementedError

    return optimizer, scheduler