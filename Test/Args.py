import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("device", type=int, default=1, help="I7-12400 + 2080Ti")
    parser.add_argument("name", type=str, default='Ben', help="My name is Ben, and this computer belong to me")
    return parser