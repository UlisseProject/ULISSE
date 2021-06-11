import sys
sys.path.append('..')

from ulissw.training import TCNTrainer
from ulissw.models import TCN
from ulissw.datasets import CustomerDataset
from ulissw.metrics import WAPE

params = {
    'MSE_WEIGHT': 3,
    'NUM_EPOCHS': 35,
    'BATCH_SIZE': 32,
    'WEIGHT_DECAY': 1e-4,
    'LR': 1e-3,
    'MOMENTUM': 0.9,
    'STEP_SIZE': 30,
    'GAMMA': 0.5,
    
    'PREPROCESS': 'minmax',

    'LOG_FREQUENCY': 1000,
    'SAVE_MODELS': True
}
folder = '../data/sonnen'

tt = TCNTrainer(params, log_dir='../logs')
tt.generate_dataset_loaders(CustomerDataset, folder=folder, 
                            prediction_interval=(192, 16), strategy='sum')
tt.generate_models_optimizers(TCN, optimizer_SGD=True, num_blocks=3, dropout=0.2)

tt.train_model(metric=[WAPE])