
from ..basic_params import parser


def get_params():
    parser.add_argument("--dylr_alpht", type=float, default=12)
    parser.add_argument("--dylr_low", type=float, default=0.00)
    parser.add_argument("--dylr_high", type=float, default=0.006)
    parser.add_argument("--task_ajust_step", type=int, default=150)
    parser.add_argument("--pre_adapt_step", type=int, default=100)
    # parser.add_argument("--gamma", type=float, default=0.01)
    # parser.add_argument("--local_attention", type=bool, default=False)
    # parser.add_argument("--resnet", type=bool, default=False)
    return parser.parse_args()
