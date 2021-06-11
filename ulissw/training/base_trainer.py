import os
from datetime import datetime
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.backends import cudnn
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer():
    '''see readme file'''
    def __init__(self, parameters, model=None, optimizer=None, scheduler=None, train_loader=None, test_loader=None, log_dir=None, **kwargs):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parameters = parameters
        self.dataset = None
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader 
        self.log_dir = log_dir
        # ---- Metrics
        self.metric_classes = None
        self.metric_names = None
        self.metric_train = None
        self.metric_test = None
        # ---- Model check
        self.save_models = False
        best_scores = None
        comparators = None
        last_best_file = None
    
    def generate_dataset_loaders(self, dataset_class, **args):
        print("[|||] INSTANTIATING DATASET [|||]")
        print(f"[|||] LOADING DATA [|||]")
        self.dataset = dataset_class(**args)

        print("[|||] PREPROCESSING DATASET [|||]")
        # Preprocessing acts in-place in the dataset object
        # Trainer class works on singe-omic dataset
        self.preprocess_dataset(self.dataset, self.parameters)

        self.train_dataset, self.test_dataset = self.dataset.train_test_split()

        self.train_loader = data.DataLoader(self.train_dataset, batch_size=self.parameters['BATCH_SIZE'], 
                                            shuffle=True, num_workers=4, 
                                            pin_memory=True, drop_last=True)
        
        self.test_loader = data.DataLoader(self.test_dataset, batch_size=16, shuffle=False, num_workers=4)
        if 'INPUT_SIZE' not in self.parameters:
            self.parameters['INPUT_SIZE'] = self.train_dataset[0][0].shape[0]
        if 'OUTPUT_SIZE' not in self.parameters:
            self.parameters['OUTPUT_SIZE'] = self.train_dataset[0][1].shape[0]


    # It is better to execute the generate_dataset_loader first in order to infer the INPUT_SIZE
    def generate_models_optimizers(self, model_class, optimizer_SGD=False, **kwargs):
        # create a model and load it to the specified device
        input_size = self.parameters.get('INPUT_SIZE', 0)
        output_size = self.parameters.get('OUTPUT_SIZE', 0)
        betas = self.parameters.get("BETAS", (0.9, 0.999))

        if input_size == 0:
            raise Exception("Either you call 'generated_datasets' first, or you have to provide INPUT_SIZE")
        print("[|||] BUILDING MODEL [|||]")

        self.model = model_class(input_size=input_size, output_size=output_size, **kwargs).to(self.device)

        # create an optimizer object
        if optimizer_SGD is False:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.parameters['LR'], 
                                        weight_decay=self.parameters['WEIGHT_DECAY'], betas=betas)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.parameters['LR'], 
                                       momentum=self.parameters['MOMENTUM'], 
                                       weight_decay=self.parameters['WEIGHT_DECAY'])

        # scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.parameters['STEP_SIZE'], 
                                                  gamma=self.parameters['GAMMA'])

    def pack_for_metric(self, metric_class, split, **kwargs):
        if split == 'train':            
            datasets = [self.train_dataset]
        elif split == 'test':
            datasets = [self.test_dataset]

        models = [self.model]
        metric = metric_class(datasets, models, self.device, **kwargs)

        return metric
    
    @staticmethod
    def preprocess_dataset(dataset, parameters):
        def minmax():
            dataset.min_max()

        option = parameters.get('PREPROCESS', None)
        if option is not None:
            preprocessor = locals()[option]
            preprocessor()

    @staticmethod
    @torch.no_grad()
    def validate_on(net, data_loader, criterion, device, is_vae=False):
        '''given a dataloader and a network performs an evaluation pass'''
        net.train(False) # Set Network to evaluation mode
        avg_loss = 0.0
        running_corrects = 0
        for input_features, _ in data_loader:
            input_features = input_features.to(device)
            
            # Forward Pass
            if is_vae:
                outputs = net(input_features)
                output = outputs[0]
            else:
                outputs = net(input_features)
            
            loss = criterion(output, input_features).item()

            avg_loss += loss                
        # Calculate Accuracy
        avg_loss = avg_loss / len(data_loader)
        
        return avg_loss

    @staticmethod
    def get_trainer_for_model(model_class):
        m_type = model_class.model_type
        trainer = None
        if m_type == 'tcn':
            from .tcn_trainer import TCNTrainer
            trainer = TCNTrainer

        return trainer

    def setup_metrics(self, metric_classes):
        self.metric_classes = metric_classes if isinstance(metric_classes, list) else [metric_classes]
        self.metric_names = [m.__name__ if m != 0 else 'mse_loss' for m in self.metric_classes]
        
        if self.metric_classes[0] != 0:
            n_metrics = len(self.metric_names)
            self.metric_train = tuple([] for _ in range(n_metrics))
            self.metric_test = tuple([] for _ in range(n_metrics))

    def eval_metrics(self, **kwargs):
        index = 0
        if (self.metric_classes[0] != 0):
            metric_test_byname = {}
            for metric in self.metric_classes:
                if metric.supports_train:
                    metric_train = self.pack_for_metric(metric_class=metric, split='train', **kwargs)
                    metric_on_train = metric_train(**kwargs)
                    self.metric_train[index].append(metric_on_train)

                metric_test = self.pack_for_metric(metric_class=metric, split='test', **kwargs)
                metric_on_test = metric_test(**kwargs)
                self.metric_test[index].append(metric_on_test)
                metric_test_byname[self.metric_names[index]] = metric_on_test

                index += 1
        self.metric_on_test = metric_test_byname

    def setup_model_saving(self, save_models):
        self.save_models = save_models
        if save_models:
            if self.log_dir is None:
                raise Exception("You asked for model saving but didn't specify log_dir")
            for name in self.metric_names:
                os.makedirs(os.path.join(self.log_dir, name), exist_ok=True)

            self.best_scores = [0 if metric != 'mse_loss' else 1e9 for metric in self.metric_names]
            self.comparators = [max if metric != 'mse_loss' else min for metric in self.metric_names]
            self.last_best_file = ["" for metric in self.metric_names]
            for i, metric in enumerate(self.metric_classes):
                if metric != 0:
                    if hasattr(metric, 'higher_better'):
                        if not metric.higher_better:
                            self.comparators[i] = min
                            self.best_scores[i] = 1e9

    def model_saving(self, avg_epoch_loss, epoch):
        if self.save_models:
            scores = [self.metric_on_test[metric] if metric != "mse_loss" else avg_epoch_loss for metric in self.metric_names]
            for i, metric in enumerate(self.metric_names):
                if self.comparators[i](self.best_scores[i], scores[i]) == scores[i]:
                    self.best_scores[i] = scores[i]
                    
                    if self.last_best_file[i] != "":
                        os.remove(self.last_best_file[i])

                    torch.save(self.model.state_dict(),
                               os.path.join(self.log_dir, metric,
                                            self.__class__.__name__ + f"epoch{epoch}.pth"))
                    self.last_best_file[i] = os.path.join(self.log_dir, metric, 
                                                          self.__class__.__name__ + f"epoch{epoch}.pth")
   

    def log_trainer(self, hparameters, train_args):
        os.makedirs(self.log_dir, exist_ok=True)
        experiment = datetime.now().strftime('%Y%m%d-%H%M%S')
        writer = SummaryWriter(log_dir=os.path.join(self.log_dir, experiment))
        hparams_all, metrics_all = {}, {}

        log_container = {}

        metric_presence = train_args.get('metric', 'no_metric')

        if isinstance(self.train_loss, list):
            stage = 0
            log_container[stage] = {}
            log_container[stage]['train_loss'] = self.train_loss
            log_container[stage]['test_loss'] = self.test_loss

            log_container[stage]['metric_names'] = self.metric_names
            if metric_presence != 'no_metric':
                log_container[stage]['metric_train'] = self.metric_train
                log_container[stage]['metric_test'] = self.metric_test

        elif isinstance(self.train_loss, tuple):
            for stage in range(len(self.train_loss)):
                log_container[stage] = {}
                log_container[stage]['train_loss'] = self.train_loss[stage]
                log_container[stage]['test_loss'] = self.test_loss[stage]
                
                log_container[stage]['metric_names'] = self.metric_names[stage]
                if metric_presence != 'no_metric':                
                    log_container[stage]['metric_train'] = self.metric_train[stage]
                    log_container[stage]['metric_test'] = self.metric_test[stage]


        n_stages = len(log_container) 
        for stage in log_container:
            metric_dict = {}

            if n_stages == 1:
                prefix_scalar = f"Stage 1"
                prefix_hparam = f"STAGE_1"
            else:
                prefix_scalar = f"part {stage+1} of {n_stages}"
                prefix_hparam = f"part_{stage+1}_of_{n_stages}"

            # Log train reconstruction loss
            for epoch, loss in enumerate(log_container[stage]['train_loss'], 1):
                if epoch == 1:
                    best = loss
                if best > loss:
                    best = loss
                writer.add_scalar(f"{prefix_scalar}/Train reconstruction loss", loss, epoch)
            metric_dict[f"{prefix_scalar}/Train reconstruction loss [best]"] = best

            # Log test reconstruction loss
            for epoch, loss in enumerate(log_container[stage]['test_loss'], 1):
                if epoch == 1:
                    best = loss
                if best > loss:
                    best = loss
                writer.add_scalar(f"{prefix_scalar}/Test reconstruction loss", loss, epoch)
            metric_dict[f"{prefix_scalar}/Test reconstruction loss [best]"] = best

            if metric_presence != 'no_metric':
                for metric_name, metric_train, metric_test in zip(log_container[stage]['metric_names'], 
                                                                  log_container[stage]['metric_train'],
                                                                  log_container[stage]['metric_test']):
                    # Log train metric
                    for epoch, metric in enumerate(metric_train, 1):
                        if epoch == 1:
                            best = metric
                        if best < metric:
                            best = metric
                        writer.add_scalar(f"{prefix_scalar}/Train metric {metric_name}", metric, epoch)
                    metric_dict[f"{prefix_scalar}/Train metric {metric_name} [best]"] = best
                    # Log test metric
                    for epoch, metric in enumerate(metric_test, 1):
                        if epoch == 1:
                            best = metric
                        if best < metric:
                            best = metric
                        writer.add_scalar(f"{prefix_scalar}/Test metric {metric_name}", metric, epoch)
                    metric_dict[f"{prefix_scalar}/Test metric {metric_name} [best]"] = best

            # --- just to avoid big names below
            metric_names = log_container[stage]['metric_names']
            # ---

            metrics_all.update(metric_dict)

        hparam_dict = {f'{prefix_hparam}_{hparam}': hparam_value for hparam, hparam_value in hparameters.items()}
        hparam_dict[f'{prefix_hparam}_metric'] = ", ".join(metric_names) if metric_presence != 'no_metric' else 'no_metric'
                
        hparams_all.update(hparam_dict)

        # hparams function must be called with the complete dictionary of all hparams of all stages
        # you cannot do hparams({'lr_stage1': 1}, {}) and then later hparams({'lr_stage2': .5}, {})
        # you must do hparams({'lr_stage1': 1, 'lr_stage2': .5}, {})
        exp, ssi, sei = hparams(hparams_all, metrics_all)

        writer.file_writer.add_summary(exp)
        writer.file_writer.add_summary(ssi)
        writer.file_writer.add_summary(sei)
        for k, v in metrics_all.items():
            writer.add_scalar(k, v)

        writer.close()