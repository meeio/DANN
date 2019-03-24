from mground.plot_utils import plot_all
from mtrain.mloger import read_step_and_loss
import numpy as np


if __name__ == "__main__":

    record_dat = read_step_and_loss(
        # train_loss = r'G:\VS Code\DANN\_MLOGS\deepCoral\lambda_10\predict.log',
        s20=r"C:\Users\meeio\Desktop\20s_mean_entropy.log",
        t20=r"C:\Users\meeio\Desktop\20t_mean_entropy.log",
        s10=r"C:\Users\meeio\Desktop\10s_mean_entropy.log",
        t10=r"C:\Users\meeio\Desktop\10t_mean_entropy.log",
    )

    # x = record_dat['t10'][0]
    # y_t = record_dat['t10'][1]
    # y_s = record_dat['s10'][1]

    x10 = record_dat["t10"][0]
    bias10 = [
        record_dat["t10"][1][i] - record_dat["s10"][1][i]
        for i in range(len(x10))
    ]

    x20 = record_dat["t20"][0]
    bias20 = [
        record_dat["t20"][1][i] - record_dat["s20"][1][i]
        for i in range(len(x20))
    ]

    a = {"bias10": (x10, bias10), "bias20": (x20, bias20)}

    plot_all(a, tagname="lambda 8")

