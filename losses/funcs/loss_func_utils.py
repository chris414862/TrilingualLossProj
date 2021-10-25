import torch


def check_tensor(tens, ident, label): 
    if torch.isnan(tens).sum() > 0 or torch.isinf(tens).sum() > 0:
        print(f"LOSS FUNC: id: {ident} label: {label} nans: {torch.isnan(tens).sum()}, infs: {torch.isinf(tens).sum()}")
        return True
    else:
        return False


def cosine_sim_matrix(view1, view2, epsilon=1e-8, debug=False):
    # view1, view2 dims: [batch,embed_size]
    view1_norm = torch.norm(view1, p=None, dim=1, keepdim=True)
    # view1_norm dims: [batch, 1]
    view2_norm = torch.norm(view2, p=None, dim=1, keepdim=True)
    # view2_norm dims: [batch, 1]
    
    dot_sim = torch.mm(view1, view2.transpose(0,1))
    # dot_sim dims: [batch,batch]
    norm_mat = torch.mm(view1_norm, view2_norm.transpose(0,1))
    # norm_mat dims: [batch, batch]

    norm_mat = torch.clamp(norm_mat, min=epsilon)
    # norm_mat dims: [batch, batch]

    cosine_sim = dot_sim / norm_mat
    # print("cosine_sim nans") if debug else ""
    # print(torch.isnan(cosine_sim).sum()) if debug else ""
    # cosine_sim dims: [batch, batch]

    return cosine_sim


def dot_sim_matrix(view1, view2, debug=False):
    S = torch.mm(view1, view2.t())
    return S



