import torch
from torch.utils import data


class WAPE:
    supports_train = True
    higher_better = False
    def __init__(self, datasets, models, device='cuda', **kwargs):
        self.dataset = datasets[0]
        self.model = models[0]
        self.device = device
        self.has_meta = self.dataset.add_month_hour
        self.predictions, self.reals = self.__gen_predictions()

    def __gen_predictions(self):
        dataloader = data.DataLoader(self.dataset, batch_size=128, shuffle=False,
                                     pin_memory=True, drop_last=False)

        preds = []
        real = []
        self.model.to(self.device)
        with torch.no_grad():
            for seq, future_seq in dataloader:
                if not self.has_meta:
                    seq = seq.unsqueeze(1).to(self.device)
                else:
                    seq = seq.to(self.device)  
                res = self.model(seq)
                preds.append(res.detach().cpu())
                real.append(future_seq)
        preds = torch.cat(preds, dim=0)
        real = torch.cat(real, dim=0)

        return preds, real

    def __call__(self, **kwargs):
        mae = (torch.abs(self.reals.flatten() - self.predictions.flatten())).mean()
        return mae/self.reals.mean()