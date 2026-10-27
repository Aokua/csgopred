# 复现说明（Reproducibility）

本目录提供论文全部实验的可复现材料。所有随机过程固定 `seed = 42`，评估严格按时间切分。

## 1. 数据快照

| 项 | 值 |
|---|---|
| 数据库 | `data/csgopred.db` |
| 快照哈希 | 见 `repro/snapshot.json` 的 `database.sha256` |
| 模型 | `data/model.pkl`，哈希见 `model.sha256` |
| 行数 | matches / maps / news / news_events 见 `counts` |

生成/校验快照：

```bash
python repro/make_snapshot.py
```

把 `snapshot.json` 里的 sha256 与本地文件对比即可确认数据一致。

## 2. 环境

```bash
python -c "import sys,numpy,sklearn;print(sys.version);print(numpy.__version__, sklearn.__version__)"
```

论文使用的版本记录在 `snapshot.json` 的 `python` / `numpy` / `sklearn` 字段。

## 3. 训练复现

```bash
export PYTHONPATH=src
python repro/train.py
```

输出：

- `repro/results.json`：两个特征集在三个时间切分上的对数损失
- `data/model_repro.pkl`：用推荐 25 维特征集重训的模型

论文表 5-1、表 5-4 的数值可直接对齐 `results.json` 与 `paper/data/*.json`。

## 4. 实验复现

| 脚本 | 产出 | 对应论文章节 |
|---|---|---|
| `experiments/decompose.py` | `paper/data/decomp.json` | 5.5 信息源贡献分解（图 5-1、5-2） |
| `experiments/feature_sets.py` | `paper/data/feature_sets.json` | 表 5-4 推荐特征集 |
| `experiments/format_revenge.py` | `paper/data/format_revenge.json` | 5.7.1 地图数建模 |
| `experiments/revenge_groups.py` | 终端输出 | 5.7.2 再相遇分组 |
| `experiments/rematch_calibration.py` | 终端输出 | 5.7.2 Elo 过度自信 |
| `experiments/make_figures.py` | `paper/images/*.png` | 图 3-1、5-1～5-4、6-1 |

依次运行：

```bash
export PYTHONPATH=src
python experiments/decompose.py
python experiments/feature_sets.py
python experiments/format_revenge.py
python experiments/revenge_groups.py
python experiments/rematch_calibration.py
python experiments/make_figures.py
```

## 5. 配置

`repro/config.json` 汇总了：

- 超参数（C=0.5、liblinear、时间权重半衰期 1500 天、校准比例 0.15）
- Elo 与地图 Elo 参数
- 三套特征列表（base20 / base+map / recommended25）
- 时间切分点

## 6. 已知限制

- 数据库为研究用快照，不含原始网页与抓取日志；
- 新闻事件由大语言模型抽取，`news_events` 表保留了原文引用片段以便核验；
- 地图数据覆盖率约 17%，未覆盖比赛的特征取中性值。
