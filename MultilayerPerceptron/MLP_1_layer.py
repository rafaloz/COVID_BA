import torch
import random

from typing import Sequence, Union, List

import numpy as np

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

from metrics_utils import *

import matplotlib.pyplot as plt

def eval_huber(model, X, y, y_span, y_lower, beta=3):
    """Devuelve Smooth-L1 (Huber) sobre un tensor completo."""
    huber = nn.SmoothL1Loss(beta=beta)
    with torch.no_grad():
        y_pred = torch.sigmoid(model(X)).squeeze() * y_span + y_lower
        loss   = huber(y, y_pred)
    return loss.item()


class Args:
    # λmax and loss flavour are the two things you will tune
    init_lambda = 0.8 # 🟡  strength of skew
    loss_type   = "L1"               #  "L1" | "L2" | "SVR"
    correlation_type = "pearson"     # only used in monitoring

class Dataset(torch.utils.data.Dataset):
    """
    Wraps (x, y, w) where w is optional.
    If w is None, __getitem__ returns (x, y).
    Otherwise it returns (x, y, w).
    """
    def __init__(self, x, y, w=None):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.w = None if w is None else torch.as_tensor(w, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.w is None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx], self.y[idx], self.w[idx]


class PerceptronFunnel(nn.Module):
    """
    Perceptrón multicapa con forma de embudo.

    Args
    ----
    input_size : int
        Dimensión del vector de entrada (nº de *features*).
    hidden_sizes : Union[int, Sequence[int]]
        • Si es una lista/tupla ⇒ tamaños explícitos de cada capa oculta
          (p.ej. [128, 64, 32]).
        • Si es un entero ⇒ tamaño inicial; el resto se genera dividiendo por
          2 hasta llegar a `min_hidden`.
    dropout_prob : float
        Probabilidad de *dropout* después de cada capa oculta.
    min_hidden : int
        Tamaño mínimo permitido al auto-generar el embudo.
    batch_norm : bool
        Inserta `nn.BatchNorm1d` tras cada capa oculta si es True.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Union[int, Sequence[int]],
        dropout_prob: float = 0.3,
        min_hidden: int = 8,
        batch_norm: bool = True,
    ):
        super().__init__()

        # ── 1. Construir la lista de tamaños ocultos ──────────────────────────────
        if isinstance(hidden_sizes, int):
            sizes: List[int] = []
            h = hidden_sizes
            while h >= min_hidden:
                sizes.append(h)
                h //= 2                       # divide por 2 en cada “piso”
            hidden_sizes = sizes

        if len(hidden_sizes) == 0:
            raise ValueError("`hidden_sizes` no puede quedar vacío.")

        # ── 2. Definir las capas ──────────────────────────────────────────────────
        layers = []
        in_features = input_size
        for out_features in hidden_sizes:
            layers.append(nn.Linear(in_features, out_features))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))
            in_features = out_features

        # Capa de salida (regresión)
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    # ── 3. Forward ────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    # ── 4. Reset de pesos opcional ───────────────────────────────────────────────
    def init_params(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


class Perceptron():
    def __init__(self, epochs=1800):
        self.model = None
        self.epochs = epochs
        self.epoch = 0

    def fit(self, Xtrain, ytrain, Xval, yval, fold, input_size, hidden_size, lr=0.01, weight_decay=1e-4, dropout=0.5,
            pretrained_model_path=None, patience=10, patience_lr=5, batch_size=16):
        X_val = (torch.tensor(Xval)).float()
        y_val = torch.tensor(yval).float()
        X = (torch.tensor(Xtrain)).float()
        y = torch.tensor(ytrain).float()

        self.y_span = max(y) - min(y)
        self.y_lower = min(y)

        if self.model is None:
            model = PerceptronFunnel(input_size=input_size, hidden_sizes=[hidden_size], dropout_prob=dropout)
            if pretrained_model_path:
                model.load_state_dict(torch.load(pretrained_model_path))
                print(f"Loaded pretrained model from {pretrained_model_path}")
            else:
                model.init_params()
            self.model = model

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        args = Args()

        # dataset statistics — calculate once from your training set
        median_age = y.median().item()
        age_min, age_max = y.min().item(), y.max().item()
        lim = (age_min, age_max)  # needed by λ(y)

        # crit = nn.L1Loss()
        # crit = SkewedLossFunction_Ordinary(args, lim, median_age)
        crit = nn.SmoothL1Loss(beta=3)
        l1_lambda = 1e-3

        best_epoch = 0
        best_mae_val = np.inf
        val_improve_epoch = 0
        total_updates = 0

        train_loss_list, val_loss_list, epoch_list = [], [], []

        # ----- before the epoch loop -----
        ages = y.numpy()
        bins = np.floor(ages).astype(int)
        counts = np.bincount(bins)

        alpha = 0.25  # 0.5 = sqrt, 0.3 = más suave
        sample_w = 1.0 / (counts[bins] ** alpha)

        # normaliza para que ⟨w⟩ = 1 (opcional)
        sample_w /= sample_w.mean()

        # probability ∝ 1 / N(age)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_w,
            num_samples=len(sample_w),  # one full epoch
            replacement=True)

        for epoch in range(self.epochs):
            model.train()
            train_dataset = Dataset(X, y)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=int(X.shape[0] / batch_size), drop_last=True)
            total_train_loss_raw = 0.0

            running_loss = 0.0  # suma de la loss *por muestra*
            running_samples = 0  # cuántas muestras llevamos

            for x_b, y_b in train_loader:  # recibe (x, y, w)
                y_pred = torch.sigmoid(model(x_b)).squeeze() * self.y_span + self.y_lower
                per_sample = crit(y_pred.reshape(-1, 1), y_b.reshape(-1, 1)).squeeze()

                l1_norm = sum(p.abs().sum() for name, p in model.named_parameters() if 'weight' in name)  # o suma sobre todos los parámetros
                loss = per_sample.mean() + l1_lambda * l1_norm  # ← nueva loss

                total_train_loss_raw += loss.item()
                # ---- acumulamos pérdida y nº de muestras ----
                running_loss += loss.item() * x_b.size(0)  # loss total del batch
                running_samples += x_b.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_updates += 1

            # Validation
            model.eval()
            with torch.no_grad():
                val_loss_eval = eval_huber(model, X_val, y_val, self.y_span, self.y_lower, beta=3)
                train_loss_eval = eval_huber(model, X, y, self.y_span, self.y_lower, beta=3)

            scheduler.step(val_loss_eval)

            # Save best model
            if val_loss_eval < best_mae_val:
                best_mae_val = val_loss_eval
                best_mae_train_eval = train_loss_eval
                best_mae_train = running_loss/running_samples
                best_model = model
                best_epoch = epoch
                val_improve_epoch = epoch

            # Early-stopping
            if epoch - val_improve_epoch >= patience:
                break

            # Logging
            epoch_list.append(epoch)
            train_loss_list.append(running_loss/running_samples)  # << eval
            val_loss_list.append(val_loss_eval)

            # print('model, epoch: {}, mae_train: {:.4f}, mae_val: {:.4f}'.format(epoch, total_train_loss / len(train_loader), mae_val))

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_list, train_loss_list, label='Training Loss', color='blue')
        plt.plot(epoch_list, val_loss_list, label='Validation Loss', color='red')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_validation_loss_' + str(fold) + '.svg')

        print(
            f"No improvement in validation in the last {patience} epochs, returning best model (epoch {best_epoch}) with best_mae_val: {best_mae_val:.4f} and mae_train: { best_mae_train:.4f}, mae_train_eval: { best_mae_train_eval:.4f}")
        return best_model


    def predict(self, X):
        X = (torch.tensor(X)).float()
        result = torch.sigmoid((self.model(X)).squeeze()) * self.y_span + self.y_lower
        prediction = result.cpu().detach().numpy()
        return prediction.squeeze()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





