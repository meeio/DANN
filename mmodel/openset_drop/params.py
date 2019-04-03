from ..basic_params import parser


def get_params():
    parser.add_argument("--dylr_alpht", type=float, default=20)
    parser.add_argument("--dylr_center", type=float, default=0.15)
    parser.add_argument("--dylr_high", type=float, default=0.15)
    parser.add_argument("--task_ajust_step", type=int, default=300)
    parser.add_argument("--pre_adapt_step", type=int, default=100)
    return parser.parse_args()
