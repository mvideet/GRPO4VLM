from llava.train.train import train


if __name__ == "__main__":
    # Do not force FlashAttention2, since it requires the optional
    # `flash_attn` package and a compatible CUDA/PyTorch build.
    # Let transformers/LLaVA pick a suitable attention implementation
    # (SDPA / eager) based on the current environment instead.
    train()
