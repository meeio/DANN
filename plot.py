
from mground.plot_utils import plot_all
from mmodel.mloger import read_step_and_loss
import numpy as np


if __name__ == "__main__":
    
    record_dat = read_step_and_loss(
        # train_loss = r'G:\VS Code\DANN\_MLOGS\deepCoral\lambda_10\predict.log',
        v_loss = r'G:\VS Code\DANN\_MLOGS\deepCoral\lambda_8\valid_loss.log',
        v_accu = r'G:\VS Code\DANN\_MLOGS\deepCoral\lambda_8\valid_acuu.log',
    )

    plot_all(record_dat, tagname='lambda 8')