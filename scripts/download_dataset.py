import argparse
from datasets import load_dataset


def preprocess_base(sample):
    return sample

PREPROCESS_FUNCTIONS = {
    "prova": preprocess_base
}


def main(args):
    ds = load_dataset(args.hf_link)

    if not args.preprocess_function is None:
        ds.map(PREPROCESS_FUNCTIONS[args.preprocess_function])

    ds.save_to_disk(args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Download and process dataset for finetuning',
                    description='brief description',
                    epilog='Text at the bottom of help')

    parser.add_argument('--hf_link', help='huggingface link')
    parser.add_argument('--save_dir', help='save directory')
    parser.add_argument('--preprocess_function', help='preprocess function', default=None)

    args = parser.parse_args()
    main(args)
