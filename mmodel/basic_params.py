import argparse
def get_param_parser():
    """get all parameters about network
    
    Returns:
        dict -- dict of parameters
    """
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size", type=int, default=50, help="Size for Mini-Batch Optimization"
    )

    parser.add_argument(
        "--eval_batch_size", type=int, default=36, help="Size for Mini-Batch Optimization"
    )

    parser.add_argument(
        "--use_gpu", type=bool, default=True, help="Use GPU to train the model"
    )

    parser.add_argument("--steps", type=int, default=4000, help="Epochs of train data")

    parser.add_argument("--log_per_step", type=int, default=20, help="Epochs of train data")

    parser.add_argument("--eval_per_step", type=int, default=100, help="Epochs of train data")


    parser.add_argument(
        "--log", type=bool, default=True, help="log record data to file or not."
    )

    parser.add_argument("--tag", type=str, default="testes", help="tag for this train.")

    parser.add_argument(
        "--ckt_path", type=str, default=None, help="Check point save path."
    )
    parser.add_argument(
        "--make_record", type=bool, default=False, help="Check point save path."
    )

    parser.add_argument(
        "--lr", type=float, default=0.01
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

    return parser

# 0.01
# A > W
# W > D
# W > A
# D > A
# 0.003
# A > D
# D > W
