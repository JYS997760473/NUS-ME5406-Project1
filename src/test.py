import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="",
                        help="this is help")
    parser.add_argument("--dataset", default="nuscenes",
                        help="this is dataset")
    parser.add_argument("--x", type=float, default=1.0)
    parser.add_argument("--t", action="store_true",
                        help="test true")
    opt = parser.parse_args()
    print(type(opt.x))