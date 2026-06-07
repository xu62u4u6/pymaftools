# pymaftools 繪圖模組檢視報告

> 日期：2026-06-01
> 方法：從使用者角度，用合成資料（長尾突變頻率 + 標準排序流程）實際繪製 OncoPlot 系列圖，
> 觀察產出並對照原始碼。重現腳本：[`scripts/demo_oncoplot.py`](scripts/demo_oncoplot.py)，
> 產出圖：`img/demo_oncoplot_*.png`。
> 此報告取代並更新 `../pymaftools_limitations.md`（該文件基於已重構的舊單檔 `OncoPlot.py`，行號已失效）。

嚴重度：🔴 會崩潰／默默產生錯誤輸出　🟠 缺常見功能、逼使用者寫 workaround　🟡 體驗／一致性。

行號對應目前的 `pymaftools/plot/OncoPlot.py`。

---

## P0 — Bug / 高風險

### 1. 🔴 `OncoPlot.default_oncoplot()` 直接崩潰

- **現象**：呼叫最基本的便利入口 `OncoPlot.default_oncoplot(table)` 立刻 `ValueError: Expected the given number of width ratios to match the number of columns of the grid`。
- **根因**：`default_oncoplot`（`OncoPlot.py:847`）預設 `width_ratios=[20, 1, 2]`（3 欄），但 `update_layout`（`:124`）寫死建立 **4 欄** GridSpec。3 ≠ 4 → 崩潰。
- **影響**：新使用者照文件第一步就壞，且因為沒有測試（見 #7）長期未被發現。
- **建議**：`default_oncoplot` 改用 4 欄 `width_ratios`（如 `[25, 1, 1, 2]`），或讓 `update_layout` 的欄數由 `len(width_ratios)` 推導（搭配 #6 的具名槽位重構一起做）。

### 7. 🔴 OncoPlot / LollipopPlot / ExpressionTablePlot / MethodsPlot 完全沒有測試

- **現象**：`tests/plot/test_plotting.py` 只覆蓋 `PivotTablePlot`（boxplot、PCA）與 `ModelPlot`。最核心、最複雜的 `OncoPlot` 一個測試都沒有 → #1 這種崩潰才會躲過 CI。
- **建議**：補 smoke test（能畫出來、軸數正確、freq/legend 內容正確），改進每一項時同步補對應測試。

---

## P1 — 缺少常見功能

### 2. 🟠 無法繪製 feature_metadata（缺 row-side 註記）

- **現象**：`feature_metadata` 有 `pathway`、`is_driver` 這類基因層級欄位，但 OncoPlot 沒有任何方法把它畫成 heatmap 左／右側的彩色條。只有 sample 層級的 `plot_categorical_metadata` / `plot_numeric_metadata`（畫在底部）。
- **根因**：版面（`update_layout`，`:110`）只規劃了底部的 sample metadata 軸（`axs_categorical_columns` / `axs_numeric_columns`），完全沒有 feature 維度的槽位。
- **建議**：新增 `plot_feature_metadata(columns, side="right", ...)`，依 `feature_metadata` 欄位在 heatmap 側邊畫 row-aligned 彩色條（categorical + numeric），並自動加入 legend。版面需多開可選的左／右欄槽位。

### 3. 🟠 數值 metadata 條沒有 colorbar / 數值刻度

- **現象**：見 `img/demo_oncoplot_metadata.png` 底部的 `age` 條——深淺代表什麼數值完全無從得知，legend 區也沒有它。
- **根因**：`plot_numeric_metadata`（`:190`）`cbar=False`，也不產生任何 legend／colorbar entry。
- **建議**：為每個 numeric metadata 欄產生小 colorbar（或在 legend 區加數值色階），可用參數開關。

### 4. 🟠 legend 顯示野生型 `False` 並列出資料中不存在的類別

- **現象**：mutation legend 第一列是 `False`（野生型），且把 `nonsynonymous` cmap 的所有類別（含 `In_Frame_Ins`、`Multi_Hit` 等資料中根本沒有的）都列出來。
- **根因**：`mutation_heatmap`（`:401`）建 legend 時只排除 `"Unknown"`，沒有排除野生型 `False`，也沒有依「資料實際出現的類別」過濾。
- **建議**：legend 預設只顯示資料中實際出現、且非野生型（`False`/`""`/`NaN`）的類別；保留參數讓使用者強制顯示完整圖例。

---

## P2 — 體驗 / 一致性

### 5. 🟡 `add_xticklabel()` 寫死 `rotation=90`、無 fontsize

- **根因**：`OncoPlot.py:696` `def add_xticklabel(self)` 無任何參數，內部寫死 `rotation=90`。
- **建議**：`add_xticklabel(fontsize=None, rotation=90)`。

### 6. 🟠 `numeric_heatmap` 把 `ax_freq` 挪用為 colorbar，且靠呼叫順序的副作用關軸

- **現象**：數值型 heatmap（CNV）有 colorbar 就沒有 freq 欄，兩者搶同一個軸；且 `numeric_heatmap` 結尾直接 `ax_legend.axis("off")`、`ax_bar.axis("off")`。
- **根因**：`numeric_heatmap`（`:821`）`cax=self.ax_freq`；`:843-844` 強制關閉其他軸。版面槽位語意不固定、靠 method 呼叫順序硬湊。
- **建議**：colorbar 用獨立／inset 軸，讓 freq 欄與 colorbar 並存；軸的開關改成宣告式（在 `set_config` 決定要哪些 component），不要靠呼叫順序的副作用。

### 8. 🟡 `update_layout()` 用 `plt.close("all")` 重建 figure（全域副作用）

- **根因**：`OncoPlot.py:122` `plt.close("all")` 後 `plt.figure(...)`。
- **影響**：無法把 OncoPlot 嵌進更大的 figure、或同頁併多張。
- **建議**：支援傳入既有 `fig` / `SubFigure`，不動全域 pyplot 狀態。

---

## 改進順序建議

| 階段 | 項目 | 理由 |
|------|------|------|
| **第一步** | #1 修 `default_oncoplot` + #7 補 OncoPlot smoke test | 低風險、止血最基本入口，先把測試網架起來 |
| **第二步** | #4 legend 過濾 + #5 `add_xticklabel` 參數 | 小改動、立即改善每張圖 |
| **第三步** | #2 feature_metadata 繪製 + #3 numeric colorbar | 你點名的核心缺口，需動版面 |
| **第四步** | #6 + #8 版面重構（具名可選槽位、不動全域 pyplot） | 影響面大，建議一次性重構 |

每一步都同步補對應測試（Rule 9：測試要驗證「為什麼」，例如 freq 欄數值正確、legend 不含野生型）。

---

## 結構層級分析

上面的 P0/P1/P2 多半是「使用症狀」。從**設計結構**看，它們其實是少數結構決策的下游：

- **S1**：同一 `BasePlot` 下混了兩種互斥範式 —— `PivotTablePlot` 是 one-shot（自建 figure、`plt.show()`、回傳 ax），`OncoPlot` 是 stateful composite（先建固定 GridSpec 具名軸、method chain 往上塗）。`save` 對 `self.fig` 的假設因此不一致。
- **S2**：版面是 **eager + imperative**。`set_config → update_layout` 在還不知道要畫什麼時就把 GridSpec 建死 → 寫死 4 欄、0 寬假欄關槽位、`default_oncoplot` 傳 3 欄崩潰，都是同一根因。正確順序應是「宣告 component → 推導版面 → 一次 render」。
- **S3**：缺 **annotation track** 抽象。sample 註記（底部彩條）是寫死的一等公民，feature 註記沒有槽位概念 —— 但資料模型（matrix + sample_metadata + feature_metadata）本是對稱的。`feature_metadata` 畫不出來是**結構性缺口**，不是漏一個 method。
- **S4**：軸開關靠 method 呼叫順序的副作用（`numeric_heatmap` 直接關 `ax_legend`/`ax_bar`），沒有集中的 `render()` 組合階段 → 順序相依、空白軸是必然。
- **S5**：Manager 模式只做一半。`ColorManager`/`LegendManager` 存在，但每個 method 自己拼 legend；`plot_pca_samples` 還自建 local manager 而非用 `self.legend_manager`。
- **S6**：入口不一致。`table.plot.xxx` 走 `PivotTablePlot`，OncoPlot 卻要 `OncoPlot(table)` 直接建，不在 `.plot` 下。
- **S7**：綁定全域 pyplot（`plt.close("all")`/`plt.show()`/`plt.tight_layout()`），擋住嵌入更大 figure 的能力。

---

## 目標架構：track-based declarative composition

核心：把「主矩陣 + 一組註記 track」當成資料模型，`add_*` 只**登記** track，最後單一 `render()` 由已登記 track 推導版面並一次畫完。

### 資料模型

```
Track (abstract)
├─ data: 對齊主矩陣某一軸的資料
├─ side: "top" | "bottom" | "left" | "right"   # left/right 對齊 features，top/bottom 對齊 samples
├─ size: float                                  # 相對厚度（取代裸 width/height_ratios）
├─ render(ax) -> None                           # 自己畫自己
└─ legend_entries() -> dict | None              # 對 LegendManager 的貢獻（唯一真相來源）

具體 track：
- MainMatrixTrack        主 heatmap（categorical 突變 / numeric CNV，二選一）
- CategoricalTrack       類別彩條（可綁 sample 或 feature 軸 → 同一類別解決 S3）
- NumericTrack           數值彩條（自帶 colorbar → 解決 P1#3）
- BarTrack               TMB 之類的長條（per-sample / per-feature）
- FreqTrack              freq 註記欄
```

### API 草案

```python
op = (OncoPlot(mutation_table)
      .main(kind="mutation")                              # MainMatrixTrack
      .add_bar("TMB", side="top")                         # BarTrack
      .add_freq(side="right")                             # FreqTrack
      .add_sample_annotation(["subtype", "sex"], side="bottom")   # CategoricalTrack×N
      .add_sample_annotation(["age"], side="bottom")              # NumericTrack
      .add_feature_annotation(["pathway"], side="left"))          # ← 新：feature 對稱可用
op.render(fig=None)        # 由已登記 track 推導 GridSpec，一次畫完；fig 可外傳
op.save("onco.png")
```

- 版面從 track 的 `side` + `size` 推導，**不再有寫死 4 欄與 0 寬假欄**（解 S2、P0#1、P2#5）。
- legend 由各 track 的 `legend_entries()` 彙整進單一 `LegendManager`（解 S5），預設只列資料實際出現、非野生型的類別（解 P1#4）。
- `render(fig=...)` 不碰全域 pyplot（解 S7）。

### 向後相容策略

保留現有公開方法（`mutation_heatmap` / `plot_freq` / `plot_bar` / `plot_categorical_metadata` / `plot_numeric_metadata` / `add_xticklabel` / `default_oncoplot`）作為**薄包裝**：內部改成「登記對應 track（+ 必要時立即 render）」，簽名不變、預設行為不變，讓既有分析腳本不受影響。新功能走新 API。

### 遷移路徑（分階段，每階段可獨立 commit + 測試）

1. **骨架**：`Track` 抽象 + `MainMatrixTrack` + 由 track 推導的 `render()`；用它重寫 `mutation_heatmap` 路徑，補 OncoPlot smoke test（解 P0#1、#7）。
2. **sample 註記 track 化**：`CategoricalTrack`/`NumericTrack`/`BarTrack`/`FreqTrack`，把現有 `plot_*` 改成薄包裝；numeric track 自帶 colorbar（解 P1#3、P2#6）。
3. **feature 註記**：`add_feature_annotation`（left/right）——S3 的核心缺口（解 P1#2）。
4. **legend 收斂 + 收尾**：legend 單一真相來源、過濾野生型（解 P1#4、S5）；`render(fig=...)` 去全域 pyplot（解 S7）；`add_xticklabel` 開參數（解 P2#5）。

---

## 2026-06-05 補充：WES plots / similarity / interaction network

### 已完成的 WES plot API

- [x] `MAF.plot.summary()` / `plot_maf_summary`
- [x] `MAF.plot.titv()` / `summarize_titv`
- [x] `MAF.plot.rainfall()`
- [x] `MAF.plot.vaf()` / `infer_vaf`
- [x] `PivotTable.plot.somatic_interactions()`
- [x] `PivotTable.plot.somatic_interactions_stats()`
- [x] `MAF.plot.compare_cohorts()`
- [x] `MAF.plot.forest()`

驗證圖輸出在 `outputs/wes_plots/`。這些圖是用 bundled example MAF / synthetic comparison 產生的檢查 artifact，不應打包進 package data。

### 銳評紀錄

- 第一版 WES 草稿的命名有 notebook shorthand 問題：
  - `s1/s2` 不清楚是 sample、source 還是 stage。
  - `n1/n2` 不清楚是 sample count 還是 mutation count。
  - `m1/m2` 不清楚是 matrix、metadata 還是 mutation counts。
  - Fisher table 的 `a/b/c/d` 在統計教科書可接受，但 library code 應改成 domain names。
- 已改成：
  - `cohort1_sample_col`
  - `cohort2_sample_col`
  - `cohort1_sample_count`
  - `cohort2_sample_count`
  - `cohort1_gene_counts`
  - `cohort2_gene_counts`
  - `both_mutated`
  - `gene1_only`
  - `gene2_only`
  - `neither_mutated`
- API 責任邊界：
  - `MAF.plot`：raw MAF-level plots，例如 summary、Ti/Tv、rainfall、VAF、cohort comparison。
  - `PivotTable.plot`：matrix-level plots，例如 oncoplot、PCA、boxplot、somatic interactions。
  - 不要讓 `PivotTable` 的 matrix plot 依賴 raw MAF 欄位。
- 已看過的 demo 圖問題：
  - `maf_summary` 初版 legend 壓到 top-gene panel；已改到底部。
  - `somatic_interactions` 初版用 signed `-log10(FDR)` 對小 example 幾乎全同色；已改用 `log2 odds ratio`，FDR 顯著只用星號標註。
  - `forest` 初版把 infinite odds ratio drop 掉；已改成畫圖時 cap 顯示值，但統計表保留 `inf`。
- 仍需補：
  - forest plot 的 confidence interval / pseudo-count policy。
  - rainfall 的 chromosome tick / boundary labels。
  - MAF summary 的 outlier handling 與 maftools-style raw Ti/Tv inset。

### `plot_similarity_panel` 設計

定位：這不是單純 heatmap；它應該是 `SimilarityMatrix` 的「完整 cohort comparison panel」。

推薦 API：

```python
fig, result = table.plot.similarity_panel(
    method="jaccard",
    group_col="subtype",
    group_order=["LUAD", "ASC", "LUSC"],
    compare_pairs=[("LUAD", "ASC"), ("ASC", "LUSC")],
    test="mannwhitney",
)
```

也應支援 precomputed matrix：

```python
fig, result = similarity_matrix.plot.panel(
    group_col="subtype",
    compare_pairs=[("LUAD", "ASC"), ("ASC", "LUSC")],
)
```

設計原則：

- 計算和繪圖分開：
  - `compute_similarity_panel(...)` 回傳 similarity matrix、group means、p-values。
  - `plot_similarity_panel(result, ...)` 只畫圖。
- 只算一次 similarity，不要每個 subplot 重算。
- 統計方法要顯式：
  - `test="mannwhitney"` 預設。
  - 後續可加 `test="permutation"`。
- 版面：
  - 左側大圖：sample x sample similarity heatmap。
  - 底部或頂部：sample group color strip。
  - 右上：group mean similarity heatmap。
  - 右下：group p-value heatmap。
- 回傳：
  - `fig`
  - `result`
    - `similarity`
    - `group_mean`
    - `p_value`
    - `pairwise_values`
- 應該先接 `SimilarityMatrix`，再用 `PivotTable.plot.similarity_panel()` 當 convenience wrapper。

### `plot_somatic_interaction_network` 設計

定位：它是 `somatic_interactions()` 的 graph view，不應重新定義統計。

推薦 API：

```python
fig, graph, stats = table.plot.somatic_interaction_network(
    top=25,
    alpha=0.05,
    interaction="both",
    layout="spring",
)
```

設計原則：

- 統計來源唯一：
  - reuse `somatic_interactions(table, top=top, alpha=alpha)`。
  - heatmap 和 network 必須使用同一份 stats。
- 節點：
  - node = gene。
  - node size = mutation frequency 或 mutated sample count。
- 邊：
  - edge = FDR-significant gene pair。
  - color = co-occurrence vs mutual exclusivity。
  - width = `-log10(adjusted_p_value)` 或 `abs(log2(odds_ratio))`。
- 回傳：
  - `fig`
  - `graph`
  - `stats`
- 依賴：
  - NetworkX 是合理選擇，但要決定是否變成 hard dependency。
  - 若不想加 dependency，先提供 stats + edge table，network plot 放 optional extra。
- 空結果：
  - 預設 fail loud：沒有 significant edges 就 raise `ValueError`。
  - 可加 `show_all=True` 當 exploratory mode，但不該預設畫一張密密麻麻的假訊號網路。

### ASC_0217 可搬用圖型

優先搬：

- `reproduce/analysis/similarity_analysis.py`
  - `plot_similarity_panel` 的最好原型。
  - 已有 sample similarity heatmap、group mean heatmap、p-value heatmap、within-vs-across boxplot。
- `scripts/WES/cooccurance.py`
  - `plot_somatic_interaction_network` 的原型。
  - 要搬抽象，不搬 notebook-style global code。
- `reproduce/analysis/mutation_profile.py`
  - DDR gene/pathway oncoplot recipe。
  - multi-metric boxplot panel recipe。

暫不搬：

- CNVkit per-sample scatter / diagram。
- Sequenza model-fit / genome-view report。
- SignatureAnalyzer WebPNGs。

這些是外部工具 QC/report 圖，不適合直接變成 pymaftools core plot API。

## 2026-06-05 House Aesthetic Spec（從 lollipop / oncoplot 反推）

**結論：pymaftools 已經有一套好看的家族美學 —— 就是 `OncoPlot` 與 `LollipopPlot`。
新圖（wes / signature）必須對齊它，而不是各自用 matplotlib/pandas/seaborn 預設。**
「美學一致」不是再發明風格，是把既有風格抽成共用層。

### 好看的參考圖（lollipop EGFR、exon-size oncoplot）為什麼好看
1. **Frameless / 無資料區格線** — 圖沒有外框、資料區沒有 grid，資料漂在留白裡。
   （wes 圖即使 despine 後仍留 spine + grid → spreadsheet 感。）
2. **Legend 是右側「有標題的卡片」** — `Domains` / `Mutation Types` / `Mutation` /
   `subtype` / `log2OR` 各自獨立、垂直堆疊、與圖分離（這是 `LegendManager` 的產物）。
   wes 用 inline matplotlib legend，擠在圖邊。
3. **單一 canonical 變異色盤,到處一樣** — 柔和、低飽和、可區分(grey/orange/blue/
   green…),lollipop 與 oncoplot 用同一套(= `ColorManager.NONSYNONYMOUS_CMAP`)。
4. **marker/形狀有設計** — lollipop 圓點有深色細邊、依 count 縮放、細灰 stem；domain
   是乾淨填色方塊。不是預設樣式。
5. **排版與留白** — 大邊距、清楚層級、標題/標籤可讀。wes 的 90° 小字 xlabel 不可讀。
6. **顏色用得有語意且節制** — 顏色編碼類別/數值,白灰為底;wes 滿版 coolwarm/viridis 太搶。

### 差距根因
wes / MafPlot **繞過 BasePlot（→ LegendManager + ColorManager）**,所以重新引入 matplotlib
預設。要美學一致,結構上就得讓它們**走同一個 legend/color 基礎建設**。

### 實作方向（house style 應該長成的樣子）
- **frameless**:全 despine + 預設不畫 grid（或極淡）;`style.py` 的 `style_axes`
  之後預設 `grid=None`、移除 left/bottom 以外所有 spine。
- **legend 走 LegendManager 的卡片樣式**:wes 圖的 legend 不要用 `ax.legend`,改用
  右側標題卡片(與 oncoplot 一致)。需要讓 wes 能取得一個 LegendManager(MafPlot
  繼承 BasePlot,或 wes 函式吃一個 color/legend context)。
- **顏色全部來自 ColorManager**:類別→`*_CMAP`;連續→house `SEQUENTIAL`/`DIVERGING`。
- **base context**:可用 `seaborn.set_theme(style="white", context="notebook")` 當底,
  但顏色/legend 仍由 ColorManager/LegendManager 覆寫 —— seaborn 只負責字級/留白基準。
- **markers**:點圖統一深色細邊 + 適度大小;heatmap 用白細格線分隔。

### 驗收（呼應 CLAUDE.md §4）
重繪後存 `outputs/design_preview/`,**並排對照 lollipop/oncoplot 人眼檢查**:legend
是否成卡片、是否 frameless、同一變異類別是否同色。看起來「像同一個庫畫的」才算過。

## 2026-06-07 樣本數敏感度盤點（哪些圖受 cohort size 影響）

對應 CLAUDE.md §4「跨樣本數量級測圖」。逐圖分三類:

### A. 高 N 版面爆掉（每樣本一個 x 元素）→ 必測 300+
| 圖 | 問題 | 現況 |
|----|------|------|
| `plot_titv` | 每樣本一根 bar + 90° 標籤 | ⚠️ 未處理,高 N 標籤糊成黑帶 |
| `SignatureTablePlot.stacked_bar` | 每樣本一根 bar | ⚠️ 222 樣本已實測標籤糊掉 |
| `plot_maf_summary` burden 格 | 每樣本一根 + 90° | ⚠️ 未處理 |
| `plot_sample_burden`(overview) | 每樣本一根 | ✅ 已 `set_xticks([])` 隱藏 |
| `OncoPlot` | 欄 = 樣本 | ✅ 已有 auto-hide(`render(show_sample_labels=None)`,>50 關閉) |

**修法(已知)**:A 類未處理的(titv / signature.stacked_bar / maf_summary)
應抄 `OncoPlot` 的 sample-label auto-hide——超過閾值就隱藏 per-sample 標籤。

### B. 低 N（<10）退化/無意義 → 必測 few
| 圖 | 退化 |
|----|------|
| `plot_gene_recurrence` | 幾乎全在 1 樣本 → 單根柱 |
| `plot_top_genes` | 全 1–2 樣本,freq% 無鑑別度 |
| `plot_cohort_comparison_forest` | 完美分離 → inf OR 爆量(3v3 時 293/309) |
| `plot_somatic_interactions` | 樣本太少 → 無顯著 pair |
| `plot_snv_spectrum` | 比例噪音大 |

B 類非 bug,是資料本質;測試只需確認**不報錯 + 誠實呈現**,不必改圖。

### C. 對樣本數不敏感（聚合 / 單樣本 / 單基因）→ 低優先
`plot_mutation_composition`(3 根聚合)、`plot_rainfall`(單樣本)、
`LollipopPlot`(單基因)、`plot_vaf`(僅低 N 稀疏)。

### 待辦
- [ ] A 類補 sample-label auto-hide(titv / signature.stacked_bar / maf_summary)
- [ ] plot 測試 parametrize across regimes;few=bundled MAF,larger=外部真實 WES MAF
