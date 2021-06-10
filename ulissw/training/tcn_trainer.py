import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.backends import cudnn

from .base_trainer import BaseTrainer


class TCNTrainer(BaseTrainer):
    '''see readme file'''
    def __init__(self, parameters, model=None, optimizer=None, scheduler=None, train_loader=None, test_loader=None, **kwargs):
        super().__init__(parameters, model, optimizer, scheduler, train_loader, test_loader)
        self.train_loss = None
        self.test_loss = None
        self.mse_loss = nn.MSELoss()      
   
    def train_model(self, **kwargs):
        print("[|||] STARTED TCN TRAINING [|||]")
        
        #--- Collect parameters ---
        NUM_EPOCHS = self.parameters.get('NUM_EPOCHS', 2)
        save_models = self.parameters.get('SAVE_MODELS', False)
        log_freq = self.parameters.get('LOG_FREQUENCY', 5e3)
        metric_classes = kwargs.get('metric', 0)
        eval_freq = kwargs.get('eval_freq', 1)

        cudnn.benchmark
        self.train_loss = []
        self.test_loss = []

        for epoch in range(NUM_EPOCHS):
            mse = 0
            for step, (sequence, future_seq) in enumerate(self.train_loader):
                sequence = sequence.unsqueeze(1).to(self.device)
                future_seq = future_seq.to(self.device)
                
                self.model.train()
                self.optimizer.zero_grad()

                # compute reconstructions
                prediction = self.model(sequence)

                # compute loss
                loss = self.mse_loss(prediction, future_seq)

                # compute accumulated gradients
                loss.backward()
                self.optimizer.step()

                # add the mini-batch training loss to epoch loss
                mse += loss.item()

                # log the step training loss
                if (step + 1) % log_freq == 0:
                    print("epoch progress: {:.3f}%, step loss = {:.4f}".format((step+1)*100/len(self.train_loader), loss))


            # compute the epoch training loss
            mse = mse / len(self.train_loader)
            # display the epoch training loss
            print("epoch : {}/{}, loss = {:.4f}".format(epoch + 1, NUM_EPOCHS, loss))
            test = 0
            #test = self.validate_on(self.model, self.test_loader, self.reconstruct_loss, self.device, is_vae=True)

            self.train_loss.append(mse)
            self.test_loss.append(test)
            self.scheduler.step()        