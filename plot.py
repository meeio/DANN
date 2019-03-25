from mground.plot_utils import plot_all
from mtrain.mloger import read_step_and_loss
import numpy as np
from mtrain.watcher import parse_losses_record, parse_watcher_dict

if __name__ == "__main__":

    d = parse_watcher_dict(r'C:\Code\MSDA\RECORDS\OPENDP_0325_1153.json')
    losses = parse_losses_record(d)

    print(losses)


    # record_dat = read_step_and_loss(
    #     # train_loss = r'G:\VS Code\DANN\_MLOGS\deepCoral\lambda_10\predict.log',
    #     BP=r"C:\Users\meeio\Desktop\19-03-24-11-46\valid_accu.log",
    #     ThresholdBP=r"C:\Users\meeio\Desktop\19-03-24-11-05\valid_accu.log",
    #     Drop_props=r"C:\Users\meeio\Desktop\19-03-24-11-05\drop_prop.log"
    # )


    # x = record_dat['Drop_props'][0]
    # y = record_dat['Drop_props'][1]
    # y = [i*100 for i in y]
    # record_dat['Drop_props'] = (x, y)

    # print(y)
    # # record_dat['Drop_props'][1] = y10

    # # x = record_dat['t10'][0]
    # # y_t = record_dat['t10'][1]
    # # y_s = record_dat['s10'][1]

    # # x10 = record_dat["t10"][0]
    # # bias10 = [
    # #     record_dat["t10"][1][i] - record_dat["s10"][1][i]
    # #     for i in range(len(x10))
    # # ]

    # # x20 = record_dat["t20"][0]
    # # bias20 = [
    # #     record_dat["t20"][1][i] - record_dat["s20"][1][i]
    # #     for i in range(len(x20))
    # # ]

    # # a = {"bias10": (x10, bias10), "bias20": (x20, bias20)}

    # plot_all(record_dat, tagname="lambda 8")

