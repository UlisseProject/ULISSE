from init_scripts import init_script
init_script()

from ulissw.training import TCNTrainer
from ulissw.models import TCN
from ulissw.datasets import CustomerDataset
from ulissw.metrics import WAPE

params = {
    # preprocessing
    'PREPROCESS': 'minmax',
    'DB_STRATEGY': 'group30_sum',
    'ADD_METADATA': True,

    # model
    'DROPOUT_RATE': .25,
    'NUM_BLOCKS': 3,
    
    # train hyperpars
    'OPTIM_SGD': True,    
    'NUM_EPOCHS': 28,
    'BATCH_SIZE': 128,
    'WEIGHT_DECAY': 1e-4,
    'LR': 1e-3,
    'MOMENTUM': 0.9,
    'STEP_SIZE': 23,
    'GAMMA': 0.5,

    # loss weight    
    'MSE_WEIGHT': 3,

    # logging
    'LOG_FREQUENCY': 5e3,
    'SAVE_MODELS': True
}
folder = 'data/sonnen'

DB_STRATEGY = params.get('DB_STRATEGY')
N_INPUTS = 3 if params.get('ADD_METADATA', False) == True else 1

print(params)
tt = TCNTrainer(params, resume='logs/group30_sum/WAPE/TCNTrainerepoch5.pth', 
                log_dir=f'logs/{DB_STRATEGY}_meta')

tt.generate_dataset_loaders(CustomerDataset, folder=folder, 
                            prediction_interval=(192, 16), 
                            strategy=DB_STRATEGY,
                            add_month_hour=params.get('ADD_METADATA'))


tt.generate_models_optimizers(TCN, optimizer_SGD=params.get('OPTIM_SGD'), 
                             num_blocks=params.get('NUM_BLOCKS'),
                             dropout=params.get('DROPOUT_RATE'),
                             num_inputs=N_INPUTS)

tt.train_model(metric=[WAPE])