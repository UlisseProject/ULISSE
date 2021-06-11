from ulissw.models import TCN
from ulissw.datasets import CustomerDataset
from ulissw.utils import plot_n_sequences

d = CustomerDataset('data/sonnen', prediction_interval=(192, 16), strategy='sum')

tcn = TCN(input_size=192, num_inputs=1, output_size=16,
          num_blocks=2, n_channels=128)
tcn.load_state_dict(torch.load('logs/WAPE/TCNTrainerepoch35.pth'))


plot_n_sequences(d.db, tcn, 15, offset=10, transformer=d.revert_preprocessing, 
                 title='Prediction of total demand', save='imgs/test1.png')