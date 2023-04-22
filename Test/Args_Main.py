from Args import config_parser

parser = config_parser()
args = parser.parse_args()

if __name__=='__main__':
    print(parser.format_values())