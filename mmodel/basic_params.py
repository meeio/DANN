import argparse
from mtrain.watcher import watcher


basic_params = None
class _ParamParser(argparse.ArgumentParser):
    def parse_args(self):
        arg = super().parse_args()
        watcher.parameter_note(arg)
        global basic_params
        basic_params = arg
        return arg

parser = _ParamParser()

parser.add_argument(
    "-r", action="store_true", dest="make_record", help="Check point save path."
)

parser.add_argument(
    "--batch_size", type=int, default=50, help="Size for Mini-Batch Optimization"
)

parser.add_argument(
    "--eval_batch_size", type=int, default=36, help="Size for Mini-Batch Optimization"
)

parser.add_argument(
    "--use_gpu", type=bool, default=True, help="Use GPU to train the model"
)

parser.add_argument("--steps", type=int, default=5000, help="Epochs of train data")

parser.add_argument("--log_per_step", type=int, default=20, help="Epochs of train data")

parser.add_argument("--eval_per_step", type=int, default=100, help="Epochs of train data")


parser.add_argument("--tag", type=str, default=None, help="tag for this train.")

parser.add_argument(
    "--lr", type=float, default=0.001
)

parser.add_argument(
    "--dataset", type=str, default="Office", help="data base we will use"
)

parser.add_argument(
    "--source", type=str, default="A", help="data base we will use"
)

parser.add_argument(
    "--target", type=str, default="W", help="data base we will use"
)



# 0.01
# A > W
# W > D
# W > A
# D > A
# 0.003
# A > D
# D > W
