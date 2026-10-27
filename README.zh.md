# csgopred · CS2 胜率与概率校准模型

给定两支战队、赛制与比赛时间，输出**校准过的胜率**、**理论价格**、**参考价**与驱动因素。

当前状态：**M1 基线完成**（Elo + 逻辑回归 + walk-forward 回测），核心管线零第三方依赖；
HLTV 抓取通道已打通（结果 / 逐图比分 / veto），但需要慢速回填。

---

## 快速开始

```bash
# 0) 让 python 找到包（或 pip install -e .）
$env:PYTHONPATH="src"       # PowerShell
set PYTHONPATH=src          # CMD

# 1) 导入开源赛果（无需鉴权）
python -m csgopred ingest --path data/raw/github/csgo-dataset/csgoresults.csv --source hojlund

# 1b) 导入 Kaggle 数据集（逐图比分 / veto / 选手 / 当时排名，需 Kaggle token）
python -m csgopred ingest-kaggle --dir data/raw/kaggle

# 2) 数据概览
python -m csgopred stats

# 3) 回测
python -m csgopred backtest --split 0.7 --date-to 2023-05-08 --report M1_baseline.md

# 4) 实力排名 / 预测
python -m csgopred rank --top 20
python -m csgopred predict -a "Vitality" -b "NAVI" --format bo3
python -m csgopred predict -a "FaZe Clan" -b "G2 Esports" --format bo3 --json
python -m csgopred predict -a "Vitality" -b "NAVI" --date 2023-05-01   # 按当时信息
```

### 补 CS2 数据（HLTV）

```bash
pip install curl_cffi beautifulsoup4 lxml

# 单批：抓 20 页结果 + 50 场详情（逐图比分/veto）
python -m csgopred hltv --max-pages 20 --since 2023-09-01 --details 50 --rate 6

# 无人值守回填（推荐）：分批抓，被 Cloudflare 拦就指数冷却重试，直到 2023-09-01
scripts\run_backfill.cmd
# 日志：reports\hltv_backfill.log；停止：taskkill /IM python.exe /F
# 或直接调 CLI：
python -m csgopred hltv-backfill --since 2023-09-01 --pages 20 --rate 8 --pause 120 --cooldown 900
```

被抓取拦截时不要硬刚：`hltv-backfill` 会自动冷却；中断后带 `--offset` 续抓，缓存会跳过已下载页面。

---

## 当前基线（真实回测数字）

三个数据源合并后：**69,509 场**（去重 68,540），覆盖 **2015-11 ~ 2026-10**。

| 来源 | 场次 | 时间范围 | 提供什么 |
|---|---:|---|---|
| hojlund（GitHub） | 16,497 | 2019-07 ~ 2023-05 | 赛果 |
| HLTV（自抓） | 31,413 | 2023-08 ~ 2026-10 | 赛果 + 赛事星级 |
| Kaggle | 21,599 | 2015-11 ~ 2020-03 | 赛果 + 逐图 + veto + 选手 rating + 当时排名 |

walk-forward（训练 2015-11 ~ 2024-11，测试 2024-11 ~ 2026-10，20,562 场）：

| 模型 | log loss | Brier | 准确率 | ECE |
|---|---:|---:|---:|---:|
| **Elo + 逻辑回归** | **0.6340** | 0.2219 | 63.4% | 0.86% |
| Elo 原始映射（scale=400） | 0.6350 | 0.2224 | 63.2% | 2.30% |
| 全猜 0.5 | 0.6931 | 0.2500 | 56.1% | 6.06% |

95% bootstrap CI `[0.6297, 0.6383]`；2026 年分段 **0.6326**（9,996 场）。完整表见
[reports/M2_merged.md](reports/M2_merged.md)。

拟合出的斜率 `b ≈ 2.38`（而不是 Elo 理论值 1）说明：CS 的强弱差距比棋类更陡，
直接用 400 分基准会系统性低估强队——这也是「Elo 原始映射」校准更差的原因。

> 参考：0.634 大致是「比瞎猜好 8.5%」。真正要追的对手是**市场收盘价**（约 0.60–0.63）。

## M2：特征模型（2026-10-27）

在 Elo 之上加入 20 维特征（近况、休息天数、比赛密度、H2H、赛制、赛事级别、LAN、年代），
用按时间切分的对比评估：

| 测试起点 | 模型 | log loss | Brier | 准确率 | ECE |
|---|---|---:|---:|---:|---:|
| 2023-01 | Elo（M1） | 0.6367 | 0.2233 | 63.3% | 0.53% |
| | **LR + 特征** | **0.6303** | 0.2204 | 64.0% | 1.62% |
| | HGB + 特征 | 0.6370 | 0.2232 | 63.5% | 1.66% |
| 2025-08 | Elo（M1） | 0.6300 | 0.2201 | 64.4% | 0.87% |
| | **LR + 特征** | **0.6224** | 0.2169 | **65.1%** | 1.04% |
| | HGB + 特征 | 0.6279 | 0.2194 | 63.9% | 0.61% |

平均 log loss：Elo 0.6334 → **LR 0.6263**（−0.0071，约 1.1%），HGB 0.6324。

```bash
python -m csgopred train --splits 2023-01-01,2025-08-01 --report M2_features.md
python -m csgopred predict -a "Vitality" -b "Spirit" --format bo3 --event "IEM Katowice 2026"
```

`train` 会把全量重训的模型写到 `data/model.pkl`，`predict` 自动加载它；没有模型文件时回退 Elo。

**三个结论**：

1. **H2H 是除 Elo 外最强的特征**——去掉它 log loss 从 0.6281 掉到 0.6292。
2. **GBDT 打不过线性模型**（0.6324 vs 0.6263）：特征本身近似线性，而树模型在 6.8 万样本上更容易过拟合噪声；
   调参后（depth=3、leaf=200）也只能追平。
3. **Elo 超参比模型结构更重要**：K×1.6、净胜分加成×3、样本半衰期 1500 天，让同样特征下的 log loss 再降 0.0017。
   合并大小写变体队名（Falcons/FALCONS 等 95 个）也修掉了一批「同一支队算成两支」的问题。

### 数据现状（2026-10-27）

- 赛果时间线连续，无断档（昨晚 HLTV 回填 31,413 场，补齐 2023-08 之后）。
- 逐图比分 **42,281 条 / 24,372 场**，veto **108,702 条 / 15,532 场**——但**全部集中在 2015–2020**。
- 2023 年之后的逐图/veto 依赖 HLTV 详情页，正在被 Cloudflare 限流，抓取器在冷却重试中。

---

## 目录

```
src/csgopred/
├─ db.py                  SQLite schema / 迁移（matches, maps, veto, teams）
├─ normalize.py           队名归一、赛制解析
├─ ingest/
│  ├─ csv_source.py       开源 CSV 导入（列名自动识别）
│  ├─ kaggle_source.py    Kaggle 导入：逐图 / veto / 选手 / 排名
│  └─ hltv.py             HLTV 抓取：结果 / 地图比分 / veto
├─ features/elo.py        队伍级 Elo（不活跃衰减、赛事分级 K、净胜图加成）
├─ models/
│  ├─ rating.py           Elo 差 → 概率（Newton 法拟合逻辑回归）
│  └─ odds.py             理论价格 / 摩擦 / 分数凯利
├─ eval/
│  ├─ metrics.py          log loss / Brier / ECE / bootstrap CI
│  └─ backtest.py         walk-forward 回测 + 多源去重
└─ serve/cli.py           命令行入口
```

## 关键设计

- **零依赖核心**：回测与预测只用标准库，SQLite 存储（`data/csgopred.db`）。抓取才需要 `curl_cffi`。
- **防泄漏**：按时间切分；测试期每场先预测再更新；`--date` 只回放该日期之前的比赛。
- **多源去重**：同一场可能同时来自 CSV 与 HLTV，按「日期 + 双方队名」合并，优先 HLTV（它带地图/veto）。
- **不活跃衰减**：60 天宽限，之后半衰期 180 天回归 1500。
- **价格**：理论价 `1/p`；参考价 `q = p(1+m)`；资金配置用分数凯利（默认 1/4）。
- **样本不足时扩区间**：`predict` 用评级不确定度做蒙特卡洛，低样本队伍自动变宽。


## M4：市场对齐与评估（2026-10-27）

代码已完成，缺的是**真实收盘价格**。

```bash
python -m csgopred market --odds data/odds.csv          # 导入
python -m csgopred market --eval --start 2026-09-01     # 评估
```

CSV 格式：`date,team_a,team_b,odds_a,odds_b[,book,is_closing]`

输出：模型与市场（去水后）的 log loss / Brier、CLV 均值与正 CLV 比例、
以及阈值 0/3%/5% 下的分数凯利收益指标。

价格来源实测：Kaggle 无 CS 价格数据集；OddsPortal 是 Next.js 流式渲染、
抓不到数据接口；bo3.gg 赔率默认隐藏；Pinnacle / The Odds API 需付费或 key。


## M5：可视化界面与每日更新（2026-10-27）

### 一键启动

双击文件夹里的 **启动界面.lnk**（或 `启动界面.bat`）：

- 服务没在跑 → 自动后台启动并打开浏览器
- 服务已在跑 → 直接打开浏览器，不会重复启动
- 启动失败会提示手动命令

端口 8080，地址 http://127.0.0.1:8080



### 启动界面

```bash
python -m csgopred serve --port 8080
# 打开 http://127.0.0.1:8080
```

界面包含：比赛预测（胜率、95% 区间、理论/参考价、地图推演、驱动因素）、Elo 实力榜、
最近新闻事件、数据概览。前端为原生 HTML/JS，零依赖；后端用标准库 `http.server`。

API：
- `GET /api/predict?a=Vitality&b=Spirit&format=bo3[&date=YYYY-MM-DD]`
- `GET /api/teams?top=60`
- `GET /api/stats`
- `GET /api/news?limit=20`

### 每日更新

```bash
python -m csgopred daily --pages 2 --min-new 1
```

流程：抓 HLTV 最新赛果 → 有新增则重训 → 写 `reports/daily_YYYY-MM-DD.md`。
已注册 Windows 计划任务 `csgopred-daily`（每天 09:00）：

```bash
schtasks /Query /TN csgopred-daily
schtasks /Delete /TN csgopred-daily /F
```


## M6：地图级 Elo 与 veto 模拟（2026-10-27）

数据：Kaggle `ektarr/counter-strike-pro-matches` + 2015–2020 HLTV，
共 56,730 张地图，其中 2023+ 有 14,449 张（5,580 场）。

特征：
- 单图 Elo：每队每张图独立评分，样本不足 5 场时向队伍 Elo 收缩
- veto 模拟：按「相对强弱」ban 弱图 / pick 强图，得到实际会打的图
- 系列赛概率：按选图逐图 DP（BO1/BO3/BO5）

| 切分 | 基线 LR | + 地图特征 | 变化 |
|---|---:|---:|---:|
| 2025-08 | 0.62235 | 0.62155 | −0.00080 |
| 2026-09 | 0.62163 | 0.62057 | −0.00106 |

平均 log loss 0.6220 → 0.6211，模型 27 维，准确率 65.07% / 65.37%。

## 已知限制

1. **地图/veto 未启用**：按当前决定放弃 2015–2020 的地图数据（版本差异大），等 CS2 时代数据补齐再进模型。
2. **无阵容/新闻**：换人、伤病、教练变动的影响尚未建模（M3 的核心）。
3. **超参在评估切分上调过**：K/净胜分/半衰期是在 2023 与 2025 两个切分上网格搜索的，存在轻微乐观偏差；
   严格做法是嵌套时间交叉验证。
4. **bo1 比分口径不一**：CSV 存回合比分（16-10），HLTV/Kaggle 存系列赛比分（1-0）；只影响净胜分权重，不影响胜负。

## 下一步

- **M3**：新闻抓取 + LLM 结构化抽取（JSON schema）+ 事件特征消融——目前唯一还没动的信号源。
- **M4**：接入市场收盘价格，做 CLV / 收益评估。

规划见 [docs/技术规划.md](docs/技术规划.md)，数据源见 [docs/数据源.md](docs/数据源.md)。

## 复现与投稿

- 数据快照：`data/csgopred.db` sha256 `502f3b921afdad32…`（完整值见 `repro/snapshot.json`）
- 模型快照：`data/model.pkl` sha256 `0f1294558731015c…`
- 行数：matches 69,509（导入时去重后 68,523）、maps 56,730、news 1,097
- 复现：`python repro/make_snapshot.py` 校验哈希，`python repro/train.py` 重训，`experiments/*.py` 复现图表
- 环境：Python 3.14.6 / numpy 2.4.6 / scikit-learn 1.9.0，seed = 42

投稿目标：*Journal of Statistics and Data Science in Sports*（JSDSS，钻石开放获取，不收版面费，范围含电竞）。
