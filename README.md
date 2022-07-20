
# Inpainting at Modern Camera Resolution by Guided PatchMatch with Auto-Curation

<img src="https://github.com/owenzlz/SuperCAF/blob/main/supercaf.png" style="width:1000px;">

This repository contains the code and model for the curation module as well as test dataset described in our paper:
Inpainting at Modern Camera Resolution by Guided PatchMatch with Auto-Curation by Lingzhi Zhang et al.



## High-Resolution Dataset

You can use the command below to download our high-resolution testset (images and masks) used in this paper. 

- Download our datasets
```bash
bash download_datasets.sh
```


## Curation Module

See curator_test_torchscript.py for example code. It is licensed by Adobe for non-commercial
use only: see the below license.

An example of running on the provided images (the output should match the below output):

```bash
python curator_test_torchscript.py a.png b.png mask.png
```

Curator preferences for (a, b), higher is better:  tensor([ 0.4453, -0.4453], device='cuda:0', grad_fn=<SubBackward0>)

If you compare with this in a scientific publication then please cite the below citation.

## BibTex Citation

```bash
@InProceedings{Zhang_2022_guided_pm,
author = {Zhang, Lingzhi and Barnes, Connelly and Amirghodsi, Sohrab and Wampler, Kevin and Shechtman, Eli and Lin, Zhe and Shi, Jianbo},
title = {Inpainting at Modern Camera Resolution by Guided PatchMatch with Auto-Curation},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
month = {October},
year = {2022}
}
```

## License

Adobe Research License Terms For Redistributable Adobe Materials

```bash
1. You may use, reproduce, modify, and display the research materials provided under this license (the
"Research Materials") solely for noncommercial purposes. Noncommercial purposes include academic
research, teaching, and testing, but do not include commercial licensing or distribution, development of
commercial products, or any other activity which results in commercial gain. You may not redistribute the
Research Materials.

2. You agree to (a) comply with all laws and regulations applicable to your use of the Research Materials under
this license, including but not limited to any import or export laws; (b) preserve any copyright or other
notices from the Research Materials; and (c) for any Research Materials in object code, not attempt to
modify, reverse engineer, or decompile such Research Materials except as permitted by applicable law.

3. THE RESEARCH MATERIALS ARE PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, AND YOU ASSUME
ALL RISKS ASSOCIATED WITH THEIR USE. IN NO EVENT WILL ANYONE BE LIABLE TO YOU FOR ANY ACTUAL,
INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN CONNECTION WITH USE OF
THE RESEARCH MATERIALS.
```
