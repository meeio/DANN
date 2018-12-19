import argparse
def get_param_parser():
    """get all parameters about network
    
    Returns:
        dict -- dict of parameters
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nf",
        type=int,
        default=64,
        help="Number of filters to use in the generator network",
    )

    parser.add_argument(
        "--batch_size", type=int, default=128, help="Size for Mini-Batch Optimization"
    )

    parser.add_argument(
        "--gray", type=bool, default=True, help="Size for Mini-Batch Optimization"
    )

    parser.add_argument(
        "--class_num", type=int, default=10, help="Class number of data set"
    )

    parser.add_argument(
        "--use_gpu", type=bool, default=True, help="Use GPU to train the model"
    )


    parser.add_argument("--steps", type=int, default=150000, help="Epochs of train data")

    parser.add_argument("--log_per_step", type=int, default=100, help="Epochs of train data")

    parser.add_argument("--eval_per_step", type=int, default=500, help="Epochs of train data")

    parser.add_argument(
        "--sdsname", type=str, default="SVHN", help="data base we will use"
    )

    parser.add_argument(
        "--tdsname", type=str, default="MNIST", help="data base we will use"
    )

    parser.add_argument(
        "--log", type=bool, default=True, help="log record data to file or not."
    )

    parser.add_argument("--tag", type=str, default="testes", help="tag for this train.")

    parser.add_argument(
        "--ckt_path", type=str, default=None, help="Check point save path."
    )

    parser.add_argument(
        "--step_size", type=int, default=5, help="Step size for decaying lr."
    )

    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Gamma for decaying lr."
    )

    return parser