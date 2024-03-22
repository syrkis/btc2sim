# main.py
#   c2sim main
# by: Noah Syrkis

# imports
from src import parse_args, scripts


# main
def main():
    args = parse_args()

    if args.script in scripts:
        scripts[args.script]()


if __name__ == "__main__":
    main()
