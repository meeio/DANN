
from ..basic_params import parser


def get_params():
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--local_attention", type=bool, default=False)
    parser.add_argument("--resnet", type=bool, default=False)
    return parser.parse_args()
