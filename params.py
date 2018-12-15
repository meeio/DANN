import argparse
def get_params():
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
        "--batch_size", type=int, default=64, help="Size for Mini-Batch Optimization"
    )

  
    parser.add_argument(
        "--class_num", type=int, default=65, help="Class number of data set"
    )

    parser.add_argument(
        "--bottle_neck", type=int, default=256, help="Class number of data set"
    )

    parser.add_argument(
        "--use_gpu", type=bool, default=True, help="Use GPU to train the model"
    )

    parser.add_argument("--steps", type=int, default=2000, help="Epochs of train data")

    parser.add_argument("--log_per_step", type=int, default=50, help="Epochs of train data")

    parser.add_argument("--eval_per_step", type=int, default=150, help="Epochs of train data")

    parser.add_argument(
        "--sdsname", type=str, default="Ar", help="data base we will use"
    )

    parser.add_argument(
        "--tdsname", type=str, default="Pr", help="data base we will use"
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
        "--lr", type=float, default=0.01, help="Gamma for decaying lr."
    )

    parser.add_argument(
        "--gamma", type=float, default=0.6, help="Gamma for decaying lr."
    )

    parser.add_argument(
        "--sigma", type=float, default=10, help="Gamma for decaying lr."
    )

    parser.add_argument(
        "--hidden_size", type=int, default=1024, help="Gamma for decaying lr."
    )

    return parser.parse_args()