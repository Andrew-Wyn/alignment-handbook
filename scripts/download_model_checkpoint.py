import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.hf_link)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_link)

    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Download model checkpoint for finetuning',
                    description='brief description',
                    epilog='Text at the bottom of help')

    parser.add_argument('--hf_link', help='huggingface link')
    parser.add_argument('--save_dir', help='save directory')

    args = parser.parse_args()
    main(args)
