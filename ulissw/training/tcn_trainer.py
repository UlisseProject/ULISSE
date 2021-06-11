import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.backends import cudnn

from .base_trainer import BaseTrainer


class TCNTrainer(BaseTrainer):
    '''see readme file'''
    def __init__(self, parameters, model=None, optimizer=None, scheduler=None, train_loader=None, test_loader=None, **kwargs):
        super().__init__(parameters, model, optimizer, scheduler, train_loader, test_loader, **kwargs)
        self.train_loss = None
        self.test_loss = None
        self.mse_loss = nn.MSELoss()      
   
    def train_model(self, **kwargs):
        print("[|||] STARTED TCN TRAINING [|||]")
        
        #--- Collect parameters ---
        NUM_EPOCHS = self.parameters.get('NUM_EPOCHS', 2)
        MSE_WEIGHT = self.parameters.get('MSE_WEIGHT', 1)
        save_models = self.parameters.get('SAVE_MODELS', False)
        log_freq = self.parameters.get('LOG_FREQUENCY', 5e4)
        metric_classes = kwargs.get('metric', 0)
        eval_freq = kwargs.get('eval_freq', 1)

        # --- Log losses ---
        cudnn.benchmark
        self.train_loss = []
        self.test_loss = []

        # --- Set up metrics ---
        self.setup_metrics(metric_classes)
        
        # --- Set up model saving ---
        self.setup_model_saving(save_models)


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
                loss = MSE_WEIGHT*self.mse_loss(prediction, future_seq)

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
            print("epoch : {}/{}, loss = {:.4f}, LR = {:.4f}".format(epoch + 1, NUM_EPOCHS, mse,
                                                                    self.scheduler.get_last_lr()[0]))

            test = 0
            #test = self.validate_on(self.model, self.test_loader, self.reconstruct_loss, self.device, is_vae=True)

            # --- Check metrics and save models
            if (epoch % eval_freq == 0):
                self.eval_metrics(**kwargs)
                self.model_saving(mse, epoch+1)

            self.train_loss.append(mse)
            self.test_loss.append(test)
            self.scheduler.step()

        if self.log_dir is not None:
            self.log_trainer(self.parameters, kwargs)