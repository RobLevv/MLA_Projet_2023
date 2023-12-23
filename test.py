import matplotlib.pyplot as plt
import numpy as np

img = plt.imread('data/Img_Processed/000001.jpg')
print(np.min(img), np.max(img))

print(np.mean(img))
plt.imshow(img)
plt.show()