from trainer import Trainer
from infenrence import BeamSearcher
import config
import argparse
import warnings


def main(args):
    if config.train:
        trainer = Trainer(args)
        trainer.train()
    else:
        beamsearcher = BeamSearcher(args.model_path, args.output_dir)
        beamsearcher.decode()



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model_path", type=str, default="",
                        help="path to the saved checkpoint")
    parser.add_argument("--output_dir", type=str, default="./result/")
    args = parser.parse_args()
    main(args)
