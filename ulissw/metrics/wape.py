import torch
from torch.utils import data


class WAPE:
    supports_train = True
    higher_better = False
    def __init__(self, datasets, models, device='cuda', **kwargs):
        self.dataset = datasets[0]
        self.model = models[0]
        self.device = device
        self.predictions, self.reals = self.__gen_predictions()

    def __gen_predictions(self):
        dataloader = data.DataLoader(self.dataset, batch_size=128, shuffle=False,
                                     pin_memory=True, drop_last=False)

        preds = []
        real = []
        self.model.to(self.device)
        with torch.no_grad():
            for seq, future_seq in dataloader:
                res = self.model(seq.unsqueeze(1).to(self.device))
                preds.append(res.detach().cpu())
                real.append(future_seq)
        preds = torch.cat(preds, dim=0)
        real = torch.cat(real, dim=0)

        return preds, real

    def __call__(self, **kwargs):
        mae = (torch.abs(self.reals.flatten() - self.predictions.flatten())).mean()
        return mae/self.reals.mean()