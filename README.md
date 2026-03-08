# Lung Nodule Detection

使用 3D Vision Transformer (ViT) 在 LUNA16 数据集上做肺结节检测（分类 + 3D 边框回归）。

<img src="imgs/sample.gif" width="50%" />

## 当前目录约定

本仓库最新路径约定是把数据放在项目根目录下的 `LUNA16/`，而不是旧版 `datasets/luna16/`。

`LUNA16/` 中至少应包含：

- `annotations.csv`
- `subset0` 到 `subset9`（原始 `.mhd/.raw` 文件，用于预处理）
- `subset0_npy` 到 `subset9_npy`（预处理后文件）

## 安装依赖

```bash
pip install torch transformers scikit-learn SimpleITK pillow tqdm
```

## 数据预处理（已更新为最新路径）

旧命令：

```bash
python -c "from dataset import preprocess; preprocess('datasets/luna16')"
```

新命令：

```bash
python -c "from dataset import preprocess; preprocess('LUNA16')"
```

说明：`preprocess` 会把 `subset*/` 里的 `.mhd` 转成 `subset*_npy/`，加速训练和评估阶段读取。

## 训练

```bash
python train.py
```

提示：当前 `train.py` 里仍有硬编码路径（`H:\...`）。如果你要在当前仓库直接运行，请把 `data_dir` 改成 `LUNA16` 或绝对路径。

## 评估

```bash
python eval.py
```

提示：`eval.py` 同样包含硬编码数据路径，运行前请确认：

- `model_path` 指向可用 checkpoint
- `LUNA16_Dataset(data_dir=...)` 指向你本地数据目录

## 可视化结果

<img src="imgs/roc.png" width="50%" />

Predicted Bounding Boxes:

<img src="imgs/pred_bbox_0.gif" width="22%" /><img src="imgs/pred_bbox_3.gif" width="22%" /><img src="imgs/pred_bbox_7.gif" width="22%" /><img src="imgs/pred_bbox_62.gif" width="22%" />

Ground Truth Bounding Boxes:

<img src="imgs/gt_bbox_0.gif" width="22%" /><img src="imgs/gt_bbox_3.gif" width="22%" /><img src="imgs/gt_bbox_7.gif" width="22%" /><img src="imgs/gt_bbox_62.gif" width="22%" />

更多细节可参考 `eval.ipynb`。预训练权重见：
[Huggingface](https://huggingface.co/Hiwebsun0914/LungNoduleDetection)
