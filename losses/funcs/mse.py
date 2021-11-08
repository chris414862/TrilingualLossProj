
def mse(target, pred):
    # target dims: [batch, embed_dim]
    loss = (target-pred).pow(2).sum(dim=-1).mean()
    return loss
