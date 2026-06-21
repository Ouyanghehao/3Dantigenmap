#  加载包
from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

#  配置 
FASTA_FOLDER = "./"

GENOTYPE_COLORS = {
    "G1a": "#8e7cc3",
    "G1b": "#c27ba0",
    "G1c": "#e06666",
    "G2a": "#93c47d",
    "G2b": "#6fa8dc",
    "G2c": "#ff0000",
    "new": "#FFD700"   # 金色
}
# ================================================

def read_fasta(path):
    names = []
    seqs = []
    with open(path, 'r') as f:
        seq = ''
        name = ''
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if name:
                    seqs.append(seq)
                name = line[1:]
                names.append(name)
                seq = ''
            else:
                seq += line.upper()
        if name:
            seqs.append(seq)
    return names, seqs

# 读取所有组（蛋白序列）
all_names = []
all_seqs = []
sequence_genotypes = []

# 读取G1a~G2c_protein.fas
groups = [
    "G1a_protein",
    "G1b_protein",
    "G1c_protein",
    "G2a_protein",
    "G2b_protein",
    "G2c_protein"
]

for gt_base in groups:
    fp = os.path.join(FASTA_FOLDER, f"{gt_base}.fas")
    gt_name = gt_base.replace("_protein", "")
    if os.path.exists(fp):
        print(f"✅ 读取 {gt_base}.fas 成功")
        names, seqs = read_fasta(fp)
        all_names.extend(names)
        all_seqs.extend(seqs)
        sequence_genotypes.extend([gt_name] * len(names))

# 读取new.fasta
new_fp = os.path.join(FASTA_FOLDER, "new.fasta")
new_names = []
new_seqs = []
if os.path.exists(new_fp):
    new_names, new_seqs = read_fasta(new_fp)
    print(f"✅ 读取 new.fasta 成功，{len(new_names)} 条序列")
    all_names.extend(new_names)
    all_seqs.extend(new_seqs)
    sequence_genotypes.extend(["new"] * len(new_names))
else:
    print("⚠️ 未找到 new.fasta，请检查文件名！")

print(f"\n📊 总序列数：{len(all_names)} 条")

# 合并用于MAFFT比对
with open("all_combined.fas", "w") as f:
    for n, s in zip(all_names, all_seqs):
        f.write(f">{n}\n{s}\n")

# MAFFT比对
print("\n🔄 正在进行多序列比对...")
os.system("mafft --auto all_combined.fas > all_aligned.fas")
align = AlignIO.read("all_aligned.fas", "fasta")
print(f"✅ 比对完成，总序列数：{len(align)} 条")

# 按比对后的序列名重新匹配基因型
aligned_names = [rec.id for rec in align]
final_genotypes = []
for name in aligned_names:
    if name in new_names:
        final_genotypes.append("new")
    else:
        idx = all_names.index(name)
        final_genotypes.append(sequence_genotypes[idx])

# 计算距离矩阵
cal = DistanceCalculator('identity')
dm = cal.get_distance(align)
n = len(align)
D = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        D[i, j] = dm[i, j]

# MDS 3D降维
coords = MDS(3, dissimilarity="precomputed", random_state=42).fit_transform(D)

# 1. 找到new序列的索引
new_indices = [i for i, gt in enumerate(final_genotypes) if gt == "new"]
if not new_indices:
    print("❌ 未找到new序列！")
    exit()

# 2. 计算new序列的坐标
new_center = np.mean(coords[new_indices], axis=0)
print(f"✅ new序列原始中心坐标：{new_center}")

# 3. 所有坐标减去new
coords_centered = coords - new_center
print(f"✅ new序列平移后中心坐标：{np.mean(coords_centered[new_indices], axis=0)}")

# 4. 计算所有点的坐标范围，用于设置坐标轴
all_x = coords_centered[:, 0]
all_y = coords_centered[:, 1]
all_z = coords_centered[:, 2]

# 5. 计算最大偏移量
max_range = max(np.max(np.abs(all_x)), np.max(np.abs(all_y)), np.max(np.abs(all_z)))
padding = max_range * 0.0001  # 加10%边距，避免点贴边
axis_lim = (-max_range - padding, max_range + padding)

print(f"✅ 坐标轴范围：{axis_lim}，以(0,0,0)为中心对称")

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')

# 先画new序列（放大+描边，完全不遮挡）
xs = coords_centered[new_indices, 0]
ys = coords_centered[new_indices, 1]
zs = coords_centered[new_indices, 2]
ax.scatter(xs, ys, zs, c="#FFD700", label="S-Chimeric", s=50, alpha=1.0, edgecolors='black', linewidth=1.5)
print(f"✅ 绘制 new 序列：{len(new_indices)} 个金色点")

# 再画其他基因型
for gt in ["G1a", "G1b", "G1c", "G2a", "G2b", "G2c"]:
    indices = [i for i, g in enumerate(final_genotypes) if g == gt]
    if indices:
        xs = coords_centered[indices, 0]
        ys = coords_centered[indices, 1]
        zs = coords_centered[indices, 2]
        ax.scatter(xs, ys, zs, c=GENOTYPE_COLORS[gt], label=gt, s=30, alpha=0.9)
        print(f"✅ 绘制 {gt} 型：{len(indices)} 个点")

# ===================== 【关键】 =====================
ax.set_xlim(axis_lim)
ax.set_ylim(axis_lim)
ax.set_zlim(axis_lim)
ax.set_xticks(np.linspace(axis_lim[0], axis_lim[1], 5))
ax.set_yticks(np.linspace(axis_lim[0], axis_lim[1], 5))
ax.set_zticks(np.linspace(axis_lim[0], axis_lim[1], 5))

# ==========================================
# 【新增：隐藏坐标轴上的数字，解决拥挤】
# ==========================================
ax.set_xticklabels([])  # 隐藏X轴数值
ax.set_yticklabels([])  # 隐藏Y轴数值
ax.set_zticklabels([])  # 隐藏Z轴数值

# 图表美化（论文级）
ax.set_xlabel('MDS Axis 1', fontsize=10, labelpad=0, fontweight='bold',
          fontfamily='Arial')
ax.set_ylabel('MDS Axis 2', fontsize=10, labelpad=0, fontweight='bold',
          fontfamily='Arial')
ax.set_zlabel('MDS Axis 3', fontsize=10, labelpad=0, fontweight='bold',
          fontfamily='Arial')

ax.text2D(0.6, 0.85, 'PEDV 3D Antigenic Map', 
          fontsize=10, ha='center', va='top', transform=ax.transAxes, fontweight='bold',
          fontfamily='Arial')
        
ax.legend(title='Genotype',
          bbox_to_anchor=(0.95,0.8), loc='upper left',
          prop={'size': 10, 'weight': 'bold', 'family': 'Arial'},
          title_fontproperties={'size': 10, 'weight': 'bold', 'family': 'Arial'})

ax.grid(True, alpha=0.3)

ax.xaxis.labelpad = -10
ax.yaxis.labelpad = -10
ax.zaxis.labelpad = -10

# 调整视角
ax.view_init(elev=-4, azim=-10)
plt.tight_layout()

# 保存高清图
plt.savefig("3D_map_perfect_centered.png", dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n🎉 完美完成！图片已保存为 3D_map_perfect_centered.png")
