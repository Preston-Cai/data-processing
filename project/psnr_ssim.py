"""A practice for creating the PSNR and SSIM function.\

test result:
    image_1.jpg and image_2.jpg:
      - PSNR: 3.2367228748211954 dB
      - SSIM: -0.08168971255876709

    image_1.jpg and image_1.jpg
      - PSNR: inf dB
      - SSIM: 1.0
"""

import numpy as np
from PIL import Image
import helpers as hp

test = np.array([3, 4, 5])
print(test ** 2)

f0 = Image.open('image_1.jpg')  # relative path for image 1
g0 = Image.open('image_1.jpg')   # relative path for image 2

f = np.array(f0.convert('RGB'), dtype=np.float64)
g = np.array(g0.convert('RGB'), dtype=np.float64)


def psnr(f, g):

    mse = np.mean((f - g) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def ssim(f, g):
    return hp.l(f, g) * hp.c(f, g) * hp.s(f, g)


if __name__ == "__main__":
   
    print("PSNR:", psnr(f, g), "dB")
    print("SSIM:", ssim(f, g))