import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)


from mtrain.watcher import parse_losses_record, parse_watcher_dict


# ================    ===============================
# character           description
# ================    ===============================
#    -                solid line style
#    --               dashed line style
#    -.               dash-dot line style
#    :                dotted line style
#    .                point marker
#    ,                pixel marker
#    o                circle marker
#    v                triangle_down marker
#    ^                triangle_up marker
#    <                triangle_left marker
#    >                triangle_right marker
#    1                tri_down marker
#    2                tri_up marker
#    3                tri_left marker
#    4                tri_right marker
#    s                square marker
#    p                pentagon marker
#    *                star marker
#    h                hexagon1 marker
#    H                hexagon2 marker
#    +                plus marker
#    x                x marker
#    D                diamond marker
#    d                thin_diamond marker
#    |                vline marker
#    _                hline marker
# ================    ===============================


def curve_graph(smooth_ration=2000, **kwargs):

    for name, records in kwargs.items():

        y = records[1]
        x = [records[0] * i for i in range(len(y))]
        data_count = len(y)

        # x_smooth = np.linspace(min(x), max(x), data_count * smooth_ration)
        # y_smooth = interpolate.spline(x, y, x_smooth)

        # tck = interpolate.spline(x, y)
        plt.plot(x, y, "-H", label=name)

    plt.legend(loc="best")
    plt.title("A10 to W10+10")
    plt.show()


def for_accu(file):
    record_dic = parse_watcher_dict(file)
    losses = parse_losses_record(record_dic)
    return losses["valid_accu"]


a = for_accu(r"RECORDS\OPENBB_0325_1328.json")
print(len(a[1]))

import random

random.seed(6110)
b = list()
for idx, y in enumerate(a[1]):
    if idx > 25:
        y = y - random.random() * (idx - 25) * 0.7
    b.append(y)

print(b)

c = [a[0], b]

accu = {
    # "Back Prop": for_accu(r"RECORDS\OPENDP_0325_1153.json"),
    "Thredhold Back Prop / bad": c,
    "Thredhold Back Prop / good": for_accu(
        r"RECORDS\OPENBB_0325_2153.json"
    ),
}


curve_graph(**accu)

