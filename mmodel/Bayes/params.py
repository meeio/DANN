from ..basic_params import parser


def get_params():
    parser.add_argument("--class_num", type=int, default=10)
    return parser.parse_args()
