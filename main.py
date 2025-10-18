# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "einops",
#     "jaxtyping",
#     "numpy",
#     "torch",
#     "wandb",
# ]
# ///

# PonderNet (https://arxiv.org/abs/2107.05407) trained on a parity task.

# Glossary
# b: batch size (bs)
# s: sequence length (seq_len)
# t: time steps (ponder steps)
# h: hidden dimension (hdim)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat
from jaxtyping import Float, Int

import argparse


def get_parity_batch(
    bs: int,
    seq_len: int = 64,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[Float[Tensor, "b s"], Int[Tensor, "b 1"]]:
    # per row
    k_non_zero = torch.randint(1, seq_len + 1, (bs, 1), device=device)

    # k_non_zero entries per row set to True, uniformly at random
    mask = torch.rand(bs, seq_len, device=device).argsort(dim=1) < k_non_zero

    # Only assign rand +1 or -1 to masked entries
    x = mask * (torch.randint(0, 2, (bs, seq_len), device=device) * 2 - 1)
    y = (x == 1).sum(dim=1) % 2

    x = x.to(dtype)
    y = y[:, None]

    return x, y


def gen_geo_pmf(
    lamb: Float[Tensor, "b t 1"], dim: int = 1, eps=1e-7
) -> Float[Tensor, "b t 1"]:
    # Calculates lamb_n * prod_{i=1}^{n-1} (1 - lamb_i)
    lamb = lamb.clamp(min=eps, max=1 - eps)
    log_1m_lamb = (-lamb).log1p()
    p_n = torch.exp(log_1m_lamb.cumsum(dim=dim) - log_1m_lamb + lamb.log())
    return p_n / p_n.sum(dim=dim, keepdim=True).clamp_min(eps)


def init_gru_cell_orthogonal(cell: nn.GRUCell, update_bias=1.0):
    for name, p in cell.named_parameters():
        if "weight_hh" in name:
            nn.init.orthogonal_(p)
        elif "weight_ih" in name:
            nn.init.xavier_uniform_(p)
        elif "bias" in name:
            nn.init.zeros_(p)
            # GRU gate order: r, z, n  (reset, update, new)
            # bias chunk indices: [0:h] r, [h:2h] z, [2h:3h] n
            h = p.numel() // 3
            with torch.no_grad():
                p[h : 2 * h].fill_(update_bias)


class ParityStepModel(nn.Module):
    def __init__(self, seq_len: int, h_dim: int):
        super().__init__()

        self.rnn = nn.GRUCell(input_size=seq_len, hidden_size=h_dim)
        init_gru_cell_orthogonal(self.rnn, update_bias=1.0)

        self.layer_norm = nn.LayerNorm(h_dim)

        self.output_head = nn.Linear(h_dim, 1)

        self.lambda_head = nn.Sequential(
            nn.Linear(h_dim, 1),
            nn.Sigmoid(),  # TODO maybe omit and output logits or logsigmoid
        )

    def forward(
        self, x: Float[Tensor, "b s"], h: Float[Tensor, "b h"] = None
    ) -> tuple[Float[Tensor, "b 1"], Float[Tensor, "b h"], Float[Tensor, "b 1"]]:
        if h is None:
            h = torch.zeros(
                x.size(0), self.rnn.hidden_size, device=x.device, dtype=x.dtype
            )

        h = self.layer_norm(self.rnn(x, h))
        return self.output_head(h), h, self.lambda_head(h)


def eval_forward(
    s: ParityStepModel, x: Float[Tensor, "b s"], max_ponder_steps: int
) -> Float[Tensor, "b 1"]:
    bs, seq_len = x.shape

    y_hats, lamb_hats, should_halts = [], [], []
    has_halted = torch.zeros((bs, 1), dtype=torch.bool, device=x.device)

    h = None
    for step in range(max_ponder_steps):
        y_hat, h, lamb_hat = s(x, h)

        should_halt = torch.rand_like(lamb_hat) <= lamb_hat

        y_hats.append(y_hat)
        lamb_hats.append(lamb_hat)
        should_halts.append(should_halt)

        has_halted |= should_halt
        if has_halted.all():
            break

    y_hats = rearrange(y_hats, "t b 1 -> b t 1")
    lamb_hats = rearrange(lamb_hats, "t b 1 -> b t 1")
    should_halts = rearrange(should_halts, "t b 1 -> b t 1")

    # Get index of first halt per batch element (nice trick from lucidrains/ponder-transformer)
    first_halt_ix = (
        (should_halts.cumsum(dim=1) == 0).sum(dim=1).clamp_max(max_ponder_steps - 1)
    )
    first_halt_ix = rearrange(first_halt_ix, "b 1 -> b 1 1")  # match y_hats shape

    y_hats = torch.gather(y_hats, dim=1, index=first_halt_ix)
    y_hats = rearrange(y_hats, "b 1 1 -> b 1")

    return y_hats, step + 1


@torch.no_grad()
def eval(
    s: ParityStepModel,
    steps: int,
    bs: int,
    seq_len: int,
    max_ponder_steps: int,
    device: torch.device,
):
    acc, avg_steps = 0.0, 0.0
    for _ in range(steps):
        x, y = get_parity_batch(bs, seq_len, device=device)

        y_hats, ponder_steps = eval_forward(s, x, max_ponder_steps)
        pred = (torch.sigmoid(y_hats) > 0.5).long()
        acc += (pred == y).float().mean().item()
        avg_steps += ponder_steps

    return acc / steps, avg_steps / steps


def train_loss(
    y_hats: Float[Tensor, "b t 1"],
    lamb_hats: Float[Tensor, "b t 1"],
    y: Float[Int, "b 1"],
    p_G: Float[Tensor, "b t 1"],
    beta: float = 0.01,
    eps: float = 1e-7,
):
    b, t, _ = y_hats.shape

    # Halting distribution
    p_n = gen_geo_pmf(lamb_hats, eps=eps)  # (b, t, 1)

    # Reconstruction loss
    L_rec = F.binary_cross_entropy_with_logits(
        input=rearrange(y_hats, "b t 1 -> (b t) 1"),
        target=repeat(y.float(), "b 1 -> (b t) 1", t=t),
        reduction="none",
    )
    L_rec = rearrange(L_rec, "(b t) 1 -> b t 1", t=t)
    L_rec = (L_rec * p_n).sum(1).mean()

    # Regularization loss
    # KL(p_n || p_G). Note kl_div(input, target) = KL(target || input).
    assert torch.isfinite(p_n).all(), "p_n has non-finite values"
    assert p_n.shape == p_G.shape

    L_reg = F.kl_div(
        input=p_G.clamp_min(eps).log(),
        target=p_n.clamp_min(eps),
        reduction="batchmean",
        log_target=False,
    )

    loss = L_rec + beta * L_reg
    return loss, L_rec, L_reg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--h_dim", type=int, default=128)
    parser.add_argument("--max_ponder_steps", type=int, default=None)
    parser.add_argument("--max_steps_eps", type=float, default=0.05)
    parser.add_argument("--abs_max_ponder_steps", type=int, default=100)
    parser.add_argument("--lamb_prior", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--eval_steps", type=int, default=32)
    parser.add_argument("--eps", type=float, default=1e-7)
    parser.add_argument("--wandb", action="store_true", default=False)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    max_ponder_steps = args.max_ponder_steps
    if args.max_ponder_steps is None:
        # determine max ponder steps from lamb_prior and max_steps_eps (see Sec 2.3)
        _p_G = gen_geo_pmf(
            torch.full((args.abs_max_ponder_steps,), fill_value=args.lamb_prior), dim=0
        )
        max_ponder_steps = (_p_G.cumsum(0) < (1 - args.max_steps_eps)).sum().item()
    print(f"Using max ponder steps: {max_ponder_steps}")

    # train
    bs, seq_len = args.bs, args.seq_len
    p_G = gen_geo_pmf(
        torch.full((bs, max_ponder_steps, 1), fill_value=args.lamb_prior), eps=args.eps
    ).to(device)

    s = ParityStepModel(seq_len=seq_len, h_dim=args.h_dim).to(device)
    opt = torch.optim.Adam(s.parameters(), lr=args.lr)

    if args.wandb:
        import wandb

        wandb.init(project="ponder-net-parity")
        wandb.config.update(args)
        wandb.watch(s, log="all")

    for step in range(args.steps):
        x, y = get_parity_batch(bs, seq_len, device=device)

        h = None
        y_hats, lamb_hats = [], []
        h_norms = []  # for logging

        for _ in range(max_ponder_steps):
            y_hat, h, lamb_hat = s(x, h)

            y_hats.append(y_hat)
            lamb_hats.append(lamb_hat)

            h_norms.append(h.norm().item())

        y_hats = rearrange(y_hats, "t b 1 -> b t 1")
        lamb_hats = rearrange(lamb_hats, "t b 1 -> b t 1")

        loss, rec_loss, kl_loss = train_loss(
            y_hats, lamb_hats, y, p_G, beta=args.beta, eps=args.eps
        )

        loss.backward()
        opt.step()
        opt.zero_grad()

        if step % 500 == 0:
            acc, avg_steps = eval(
                s,
                steps=args.eval_steps,
                bs=bs,
                seq_len=seq_len,
                max_ponder_steps=max_ponder_steps,
                device=device,
            )
            print(
                f"Step {step}: loss {loss.item():.4f}, acc {acc:.4f}, avg steps {avg_steps:.2f}"
            )

            if args.wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/rec_loss": rec_loss.item(),
                        "train/kl_loss": kl_loss.item(),
                        "train/h_norm_mean": sum(h_norms) / len(h_norms),
                        "train/h_norm_max": max(h_norms),
                        "train/h_norm_min": min(h_norms),
                        "train/lamb_mean": lamb_hats.mean().item(),
                        "train/lamb_max": lamb_hats.max().item(),
                        "train/lamb_min": lamb_hats.min().item(),
                        "eval/acc": acc,
                        "eval/avg_ponder_steps": avg_steps,
                    },
                    step=step,
                )
