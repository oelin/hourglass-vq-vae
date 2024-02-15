# Hourglass VQ-VAE

An Hourglass Transformer VQ-VAE architecture.

## Goal

As part of the LatentLM project, a first-stage model capable of compressing very long sequences is neccessary. We achieve this by combining [Hourglass Transformer](https://arxiv.org/abs/2110.13711) with [FSQ](https://arxiv.org/abs/2309.15505) and [Contrastive Weight Tying](https://arxiv.org/abs/2309.08351) to construct an attention-only VQ-VAE architecture.

## TODO

- [x] Linear attention.
- [ ] GQA.
- [ ] FlashAttention2 with sliding window to replace linear attention.
- [ ] Attention upsampling to replace linear upsampling. 
- [ ] (Optional) experiment with adverserial losses (Hourglass VQ-GAN).
