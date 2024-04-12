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

# def load_checkpoint(checkpoint, model, optimizer):
#     print("=> Loading checkpoint")
#     model.load_state_dict(dict([(n, p) for n, p in checkpoint['state_dict'].items()]), strict=False)
#     optimizer.load_state_dict(checkpoint["optimizer"])
#     step = checkpoint["step"]
#     return step

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Check if optimizer state dict exists in the checkpoint
    if 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError:
            print("=> Error: Loaded optimizer state dict doesn't match the size of the current optimizer's group.")
            print("=> Initializing optimizer with default parameters.")
    else:
        print("=> Warning: Optimizer state not found in the checkpoint. Initializing optimizer with default parameters.")

    step = checkpoint.get("step", 0)  # Get the value of 'step' if it exists, otherwise default to 0
    return step
