"""Stronger models for the extended benchmark (Step 3).

All functions share the calling convention of ``fit_predict_mlp`` in
``run_experiments.py``: they receive TRAINING-fold arrays plus the test-fold
inputs and return test predictions in the original target units. Nothing here
ever sees test-fold targets. Early stopping uses whole training samples held
out inside the training fold (sample-grouped), exactly like the original MLP.

Models
------
* strong_mlp          residual multimodal MLP, LayerNorm, input dropout,
                      AdamW, early stopping, small seed ensemble
* transformer_fusion  small transformer encoder over feature-group tokens
                      (image tokens, neighbour tokens, coordinate token)
* extra_trees         nonlinear tree ensemble on PCA-reduced features

These are small educational models. They are NOT OmiCLIP or STPath and make no
claim to reproduce them.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ----------------------------- hyper-parameters ---------------------------
STRONG_HIDDEN = 512
STRONG_BLOCKS = 2
STRONG_DROPOUT = 0.3
INPUT_DROPOUT = 0.1
NN_EPOCHS = 80
NN_PATIENCE = 10
NN_BATCH = 256
NN_LR = 1e-3
NN_WEIGHT_DECAY = 1e-2
NN_VAL_SAMPLES = 3

TOKENS_PER_MODALITY = 8
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
FF_DIM = 128
TF_DROPOUT = 0.2

TREE_PCA = 64
TREE_ESTIMATORS = 200
TREE_MIN_LEAF = 5


def _split_val(groups_train: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(groups_train)
    rng = np.random.default_rng(seed)
    n_val = min(NN_VAL_SAMPLES, max(1, len(unique) // 4))
    val_samples = rng.choice(unique, size=n_val, replace=False)
    val_mask = np.isin(groups_train, val_samples)
    train_mask = ~val_mask
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        train_mask = np.ones(len(groups_train), dtype=bool)
        val_mask = train_mask
    return train_mask, val_mask, val_samples


def _train_network(build_model, image_train, neighbor_train, coord_train, y_train,
                   groups_train, image_test, neighbor_test, coord_test, seed):
    """Shared training loop. Returns (test predictions, info dict)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_mask, val_mask, val_samples = _split_val(groups_train, seed)

    s_img = StandardScaler().fit(image_train[train_mask])
    s_nb = StandardScaler().fit(neighbor_train[train_mask])
    s_y = StandardScaler().fit(y_train[train_mask])

    def T(a):
        return torch.tensor(np.asarray(a, dtype=np.float32))

    tr = [T(s_img.transform(image_train[train_mask])), T(s_nb.transform(neighbor_train[train_mask])),
          T(coord_train[train_mask]), T(s_y.transform(y_train[train_mask]))]
    va = [T(s_img.transform(image_train[val_mask])), T(s_nb.transform(neighbor_train[val_mask])),
          T(coord_train[val_mask]), T(s_y.transform(y_train[val_mask]))]

    model = build_model(image_train.shape[1], neighbor_train.shape[1],
                        coord_train.shape[1], y_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=NN_LR, weight_decay=NN_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    loss_fn = nn.MSELoss()

    n = len(tr[0])
    best_val, best_state, best_epoch, wait = np.inf, None, 0, 0
    history: List[Dict[str, float]] = []
    for epoch in range(NN_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for s in range(0, n, NN_BATCH):
            b = perm[s:s + NN_BATCH]
            opt.zero_grad()
            loss = loss_fn(model(tr[0][b], tr[1][b], tr[2][b]), tr[3][b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(b)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(va[0], va[1], va[2]), va[3]).item())
        sched.step(val_loss)
        history.append({"epoch": epoch + 1, "train_loss": total / n, "val_loss": val_loss})
        if val_loss < best_val - 1e-5:
            best_val, best_epoch, wait = val_loss, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= NN_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(T(s_img.transform(image_test)), T(s_nb.transform(neighbor_test)),
                    T(coord_test)).numpy()
    info = {"best_epoch": best_epoch, "best_val_loss": best_val, "epochs_run": len(history),
            "validation_samples": [str(s) for s in val_samples], "history": history}
    return s_y.inverse_transform(out), info


# ------------------------------- strong MLP -------------------------------

def _build_strong_mlp(image_dim, neighbor_dim, coord_dim, n_out):
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                     nn.Dropout(STRONG_DROPOUT), nn.Linear(d, d))

        def forward(self, x):
            return x + self.net(x)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop_in = nn.Dropout(INPUT_DROPOUT)
            h = STRONG_HIDDEN
            self.img = nn.Sequential(nn.Linear(image_dim, h), nn.GELU())
            self.nb = nn.Sequential(nn.Linear(neighbor_dim, h // 2), nn.GELU())
            self.co = nn.Sequential(nn.Linear(coord_dim, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
            fused = h + h // 2 + 64
            self.proj = nn.Linear(fused, h)
            self.blocks = nn.Sequential(*[Block(h) for _ in range(STRONG_BLOCKS)])
            self.head = nn.Sequential(nn.LayerNorm(h), nn.Dropout(STRONG_DROPOUT), nn.Linear(h, n_out))

        def forward(self, image, neighbor, coords):
            import torch
            z = torch.cat([self.img(self.drop_in(image)), self.nb(self.drop_in(neighbor)),
                           self.co(coords)], dim=1)
            return self.head(self.blocks(self.proj(z)))

    return Net()


def fit_predict_strong_mlp(image_train, neighbor_train, coord_train, y_train, groups_train,
                           image_test, neighbor_test, coord_test, seed, n_seeds: int = 3,
                           verbose: bool = False):
    """Residual multimodal MLP; predictions averaged over ``n_seeds`` seeds."""
    preds, infos = [], []
    for k in range(n_seeds):
        p, info = _train_network(_build_strong_mlp, image_train, neighbor_train, coord_train,
                                 y_train, groups_train, image_test, neighbor_test, coord_test,
                                 seed + 1000 * k)
        preds.append(p)
        infos.append(info)
    best = min(infos, key=lambda i: i["best_val_loss"])
    out = dict(best)
    out["members"] = [{k: v for k, v in i.items() if k != "history"} for i in infos]
    out["n_seeds"] = n_seeds
    return np.mean(preds, axis=0), out


# ------------------------- transformer-style fusion -----------------------

def _build_transformer(image_dim, neighbor_dim, coord_dim, n_out):
    import torch
    import torch.nn as nn

    def chunk_size(dim):
        return int(np.ceil(dim / TOKENS_PER_MODALITY))

    ci, cn = chunk_size(image_dim), chunk_size(neighbor_dim)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_proj = nn.Linear(ci, D_MODEL)
            self.nb_proj = nn.Linear(cn, D_MODEL)
            self.co_proj = nn.Linear(coord_dim, D_MODEL)
            self.cls = nn.Parameter(torch.zeros(1, 1, D_MODEL))
            # token-type + position embeddings: CLS, image tokens, neighbour tokens, coord
            self.pos = nn.Parameter(torch.randn(1, 2 + 2 * TOKENS_PER_MODALITY, D_MODEL) * 0.02)
            layer = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, FF_DIM, TF_DROPOUT,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, N_LAYERS, enable_nested_tensor=False)
            self.drop_in = nn.Dropout(INPUT_DROPOUT)
            self.head = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Dropout(TF_DROPOUT),
                                      nn.Linear(D_MODEL, n_out))

        @staticmethod
        def _tokens(x, chunk):
            pad = chunk * TOKENS_PER_MODALITY - x.shape[1]
            if pad > 0:
                x = torch.nn.functional.pad(x, (0, pad))
            return x.view(x.shape[0], TOKENS_PER_MODALITY, chunk)

        def forward(self, image, neighbor, coords):
            b = image.shape[0]
            ti = self.img_proj(self._tokens(self.drop_in(image), ci))
            tn = self.nb_proj(self._tokens(self.drop_in(neighbor), cn))
            tc = self.co_proj(coords).unsqueeze(1)
            seq = torch.cat([self.cls.expand(b, -1, -1), ti, tn, tc], dim=1) + self.pos
            return self.head(self.enc(seq)[:, 0])

    return Net()


def fit_predict_transformer(image_train, neighbor_train, coord_train, y_train, groups_train,
                            image_test, neighbor_test, coord_test, seed, verbose: bool = False):
    """Transformer encoder over feature-group tokens (see module docstring)."""
    return _train_network(_build_transformer, image_train, neighbor_train, coord_train, y_train,
                          groups_train, image_test, neighbor_test, coord_test, seed)


# --------------------------------- trees ----------------------------------

def fit_predict_extra_trees(image_train, neighbor_train, coord_train, y_train,
                            image_test, neighbor_test, coord_test, seed):
    """ExtraTrees (native multi-output) on PCA(image+neighbour) + coordinates.

    Scaler and PCA are fitted on the training fold only.
    """
    from sklearn.ensemble import ExtraTreesRegressor

    x_tr = np.hstack([image_train, neighbor_train])
    x_te = np.hstack([image_test, neighbor_test])
    sc = StandardScaler().fit(x_tr)
    k = min(TREE_PCA, x_tr.shape[0] - 1, x_tr.shape[1])
    pca = PCA(n_components=k, random_state=seed).fit(sc.transform(x_tr))
    z_tr = np.hstack([pca.transform(sc.transform(x_tr)), coord_train])
    z_te = np.hstack([pca.transform(sc.transform(x_te)), coord_test])
    y_sc = StandardScaler().fit(y_train)
    model = ExtraTreesRegressor(n_estimators=TREE_ESTIMATORS, min_samples_leaf=TREE_MIN_LEAF,
                                max_features=0.5, n_jobs=-1, random_state=seed)
    model.fit(z_tr, y_sc.transform(y_train))
    return y_sc.inverse_transform(model.predict(z_te))
