from mground.plot_utils import plot_all
from mtrain.mloger import read_step_and_loss
import numpy as np


if __name__ == "__main__":

    record_dat = read_step_and_loss(
        # train_loss = r'G:\VS Code\DANN\_MLOGS\deepCoral\lambda_10\predict.log',
        v_loss = r'C:\Code\MSDA\_MLOGS\19-03-21-15-42\claasify.log',
        v_accu = r'C:\Code\MSDA\_MLOGS\19-03-21-15-42\valid_accu.log',
    )
    print(record_dat)

    plot_all(record_dat, tagname='lambda 8')


