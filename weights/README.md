# Final weights (fp32)

15 U-Net checkpoints used by `notebooks/final_inference.ipynb`; check with `shasum -a 256 -c SHA256SUMS`.

| Family | Files | Encoder | Input |
|---|---|---|---|
| N1 | `effb4_s1024_f0..f4.pt` | EfficientNet-B4 | whole image resized to 1024 |
| N2 | `effb4_c2048_f0..f4.pt` | EfficientNet-B4 | 1024 crops at native 2048 |
| N9' | `effv2s_c512_f0..f4.pt` | EfficientNetV2-S (`tu-tf_efficientnetv2_s.in21k_ft_in1k`) | 512 crops at native 2048 |

Main pipeline, the selected Kaggle final (`final2` in the notebook) = N1 + N9' (1:1); the alternative
(`final1`, submitted but not selected) = N1 + N2 + N9' (1:1:1). Same files as the private Kaggle dataset
`nguyenchicuong/filseg-final-weights`.
