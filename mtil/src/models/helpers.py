import copy
import os

import clip.clip as clip
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .. import datasets, templates, utils

def batch(iterable, n=64):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_datasets_text(ds, args):
    texts = []
    for d in ds:
        ref_sentences_cls = getattr(datasets, d)
        ref_sentences = ref_sentences_cls(
            None,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        ref_template = ref_sentences.template
        ref_texts = [ref_template(x) for x in ref_sentences.classnames]
        texts.extend(ref_texts)
    ret = clip.tokenize(texts).cuda()
    return ret

def merge_we(model_0, model_1, sma_count):
    params_0 = dict(model_0.named_parameters())
    for name, param_k in model_1.named_parameters():
        if name in params_0:
            param_k.data = (param_k.data * sma_count + params_0[name].data.to(param_k.device)) / (1.0 + sma_count)
    return model_1

def wise_we(model_0, model_1, sma_count, model_n, alpha=0.95):
    params_0 = dict(model_0.named_parameters())
    params_n = dict(model_n.named_parameters())
    for name, param_k in model_1.named_parameters():
        if name in params_0 and name in params_n:
            param_k.data = (
                (param_k.data * sma_count + params_0[name].data) / (1.0 + sma_count)
            ) * alpha + params_n[name].data * (1 - alpha)
    return model_1

def moving_avg(model_0, model_1, alpha=0.999):
    params_1 = dict(model_1.named_parameters())
    for name, param_q in model_0.named_parameters():
        if name in params_1:
            param_q.data = param_q.data * alpha + params_1[name].data * (1 - alpha)

def l2_loss(model, model_ref):
    device = next(model.parameters()).device
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    ref_params = dict(model_ref.named_parameters())

    for name, param_q in model.named_parameters():
        if name in ref_params and param_q.requires_grad:
            loss = loss + F.mse_loss(param_q, ref_params[name].detach(), reduction="sum")
    return loss


def virtual_vocab(length=10, n_class=1000):
    voc_len = len(clip._tokenizer.encoder)
    texts = torch.randint(0, voc_len, (n_class, length))
    start = torch.full((n_class, 1), clip._tokenizer.encoder["<start_of_text>"])
    end = torch.full((n_class, 1), clip._tokenizer.encoder["<end_of_text>"])
    zeros = torch.zeros((n_class, 75 - length), dtype=torch.long)

    texts = torch.cat([start, texts, end, zeros], dim=1)
    return texts

def distillation(t, s, T=2):
    p = F.softmax(t / T, dim=1)
    loss = F.cross_entropy(s / T, p, reduction="mean") * (T ** 2)
    return loss

def paired_loss_new(old_pred, old_true):
    T = 2
    pred_soft = F.softmax(old_pred[:, : old_true.shape[0]] / T, dim=1)
    true_soft = F.softmax(old_true[:, : old_true.shape[0]] / T, dim=1)
    loss_old = true_soft.mul(-1 * torch.log(pred_soft))
    loss_old = loss_old.sum(1)
    loss_old = loss_old.mean() * T * T
    return loss_old

def saving (args, model, iterations):
    """Save the final model checkpoint."""
    if args.save is None:
        return

    if args.we or args.we_wise:
        to_save_model = we_model
    else:
        to_save_model = model.module if hasattr(model, "module") else model

    checkpoint = {
        "iteration": iterations,
        "state_dict": to_save_model.state_dict(),
        "args": args,
    }

    path = os.path.join(args.save, f"{args.train_dataset}.pth")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Saved model to {path}")
