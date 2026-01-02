import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir('../SDv1.5')
from metrics_utils import VGGCalculator

# 路径和参数
base_style_image_path = "/userhome/wangjunke/M3S/metric_SDv1.5/style_base"
num_styles = 4

# 构建4x4矩阵
vgg_matrix = np.zeros((num_styles, num_styles))

# 加载每张图路径
style_image_paths = [os.path.join(base_style_image_path, f"{i}.png") for i in range(1, num_styles+1)]
VGG_Scorer = VGGCalculator()
# 两两计算 VGGloss
for i in range(num_styles):
    for j in range(num_styles):
        vgg_matrix[i, j] = VGG_Scorer.calculate_similarity(style_image_paths[i], style_image_paths[j])

print("VGGloss confusion matrix:\n", vgg_matrix)

# 可视化
plt.figure(figsize=(6, 5))
sns.set(font_scale=1.2)
ax = sns.heatmap(vgg_matrix, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': 'VGGloss Similarity'})
ax.set_xticklabels([f"Style {i+1}" for i in range(num_styles)])
ax.set_yticklabels([f"Style {i+1}" for i in range(num_styles)], rotation=0)
plt.tight_layout()
plt.savefig("../metric_SDv1.5/style_confusion_matrix.png", dpi=300)
#plt.show()