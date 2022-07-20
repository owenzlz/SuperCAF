
"""
Curator module test script for ECCV 2022 paper:
Inpainting at Modern Camera Resolution by Guided PatchMatch with Auto-Curation by Lingzhi Zhang et al.
If you compare with this in a scientific publication then please cite the below citation.

Given a pair of images and a mask, this code prints the antisymmetric preferences output by the MLP
shown in Fig 3: M_ij and M_ji.

Adobe Research License Terms For Redistributable Adobe Materials

1. You may use, reproduce, modify, and display the research materials provided under this license (the
“Research Materials”) solely for noncommercial purposes. Noncommercial purposes include academic
research, teaching, and testing, but do not include commercial licensing or distribution, development of
commercial products, or any other activity which results in commercial gain. You may not redistribute the
Research Materials.

2. You agree to (a) comply with all laws and regulations applicable to your use of the Research Materials under
this license, including but not limited to any import or export laws; (b) preserve any copyright or other
notices from the Research Materials; and (c) for any Research Materials in object code, not attempt to
modify, reverse engineer, or decompile such Research Materials except as permitted by applicable law.

3. THE RESEARCH MATERIALS ARE PROVIDED “AS IS,” WITHOUT WARRANTY OF ANY KIND, AND YOU ASSUME
ALL RISKS ASSOCIATED WITH THEIR USE. IN NO EVENT WILL ANYONE BE LIABLE TO YOU FOR ANY ACTUAL,
INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN CONNECTION WITH USE OF
THE RESEARCH MATERIALS.

----

BibTex:

@InProceedings{Zhang_2022_guided_pm,
author = {Zhang, Lingzhi and Barnes, Connelly and Amirghodsi, Sohrab and Wampler, Kevin and Shechtman, Eli and Lin, Zhe and Shi, Jianbo},
title = {Inpainting at Modern Camera Resolution by Guided PatchMatch with Auto-Curation},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
month = {October},
year = {2022}
}

"""

import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def imgfile2numpy(img_file, color_type = 'RGB'):
	img_np = np.array(Image.open(img_file).convert(color_type))
	return img_np

def numpy_or_string_to_tensor(I, device, transform, is_mask=False, dtype=torch.float32):
	if not is_mask:
		img_np = imgfile2numpy(I) if not isinstance(I, np.ndarray) else I
		img_tensor = transform(img_np).unsqueeze(0).to(device)
		return img_tensor
	else:
		mask_np = imgfile2numpy(I, color_type='L') if not isinstance(I, np.ndarray) else I
		mask_tensor = (numpy2tensor(mask_np, 'L', dtype=dtype)/255.0).unsqueeze(0).to(device)
		return mask_tensor

def numpy2tensor(img_np, color_type='RGB', dtype=torch.float32):
	if color_type == 'RGB':
		img_tensor = torch.from_numpy(img_np).transpose(0,2).transpose(1,2).to(dtype)
	elif color_type == 'L':
		img_tensor = torch.from_numpy(img_np).unsqueeze(2).transpose(0,2).transpose(1,2).to(dtype)
	return img_tensor

def main():
	parser = argparse.ArgumentParser(description="Test curator network on a pair of images and a mask.")
	parser.add_argument('a', type=str, help='First image')
	parser.add_argument('b', type=str, help='Second image')
	parser.add_argument('mask', type=str, help='Hole mask')
	parser.add_argument("--checkpoint", type=str, default='curator_checkpoint.pth', help="checkpoint path")
	parser.add_argument("--device", type=str, default='cuda', help='Device for model')

	args = parser.parse_args()

	model = torch.jit.load(args.checkpoint)

	transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
				])

	a = imgfile2numpy(args.a)
	b = imgfile2numpy(args.b)
	mask = imgfile2numpy(args.mask, 'L')
	a = numpy_or_string_to_tensor(a, args.device, transform)
	b = numpy_or_string_to_tensor(b, args.device, transform)
	mask = numpy_or_string_to_tensor(mask, args.device, transform, True)

	ab = torch.cat((a, b), 0)

	result = model(ab, mask)

	print('Curator preferences for (a, b), higher is better: ', result*2-1)

if __name__ == '__main__':
	main()
