import torch

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


# def load_checkpoint(checkpoint, model, optimizer):
#     print("=> Loading checkpoint")
#     model.load_state_dict(checkpoint["state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer"])
#     step = checkpoint["step"]
#     return step

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(dict([(n, p) for n, p in checkpoint['state_dict'].items()]), strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

