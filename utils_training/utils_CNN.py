import math
pi=math.pi
import os
import torch
import shutil


def save_checkpoint(state, is_best, save_path, filename):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth'))

def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print('{}not in model_state'.format(name))
            continue
        else:
            try:
                own_state[name].copy_(param)
            except RuntimeError as e:
                print(e)
                continue
    return model

def load_checkpoint(model, optimizer=None, scheduler=None, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    best_val=-1
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        try:
            start_epoch = checkpoint['epoch']
        except:
            pass
        if 'state_dict' in checkpoint.keys():
            model = load_my_state_dict(model,checkpoint['state_dict'])
            del checkpoint['state_dict']
        else:
            model = load_my_state_dict(model,checkpoint)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            del checkpoint['optimizer']

        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
            del checkpoint['scheduler']

        try:
            best_val=checkpoint['best_loss']
        except:
            best_val=-1
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, start_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
    del checkpoint
    return model, optimizer, scheduler, start_epoch, best_val


# Argument parsing
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
