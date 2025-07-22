from utils import load_encoder_hparams_and_params
from gpt2 import GPT2

def main():
    encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")

    gpt2 = GPT2(encoder, hparams, params)

    response = gpt2.prompt()

    print(response)

if __name__ == "__main__":
    main()
