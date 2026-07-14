# pymaftools 0.5.0 收尾審計

> 更新日期：2026-07-14
>
> 本文件記錄 0.5.0 收尾後的實際狀態。狀態以目前程式、測試、文件與建置結果為準，
> 不沿用舊審計中已失效的行號或未重新驗證的結論。

## 結論

0.5.0 已處理會直接影響資料正確性、持久化與發版的主要阻斷項目。HDF5 現為正式的
canonical persistence format；SQLite API 只保留相容入口並發出棄用警告。完整測試結果為
256 passed、1 skipped、10 deselected，coverage 65.19%；Ruff、Sphinx `-W`、wheel/sdist
建置、Twine 6 metadata check 與安裝後 smoke test 均通過。

目前剩餘問題主要是 API 完整性、legacy 收斂與覆蓋率，而非必須阻止 0.5.0 發布的資料
毀損問題。

## 0.5.0 已關閉項目

| 項目 | 狀態 | 實作結果 |
|---|---|---|
| Missing 被當成 present | FIXED | `to_binary_table()` 將 NA 視為 absence；連續數值必須提供 threshold。 |
| Frequency metadata 靜默錯位 | FIXED | `add_freq()` 先驗證 metadata alignment，錯位時明確失敗。 |
| Runtime `assert` validation | FIXED | 公開輸入改用明確例外，不依賴 Python optimization mode。 |
| SQLite 欄位數限制 | FIXED | README、Skill、API docs 與 bundled example 全面採 HDF5；SQLite 已棄用。 |
| HDF5 table name/key 耦合 | FIXED | 儲存 key 與使用者 table name 解耦，並保留 metadata。 |
| LLM helper 放在 clustering core | FIXED | 搬至 `pymaftools.llm`，client 由呼叫端注入，無 OpenAI hard dependency。 |
| MAF schema 與 sample ordering | FIXED | 讀取時驗證必要欄位並保留明確樣本順序。 |
| TCGA builder 不完整輸入 | FIXED | 補空目錄、重複 case、缺檔與 timeout 等邊界處理。 |
| VCF 公開入口與 multi-allelic parsing | FIXED | `VCF.read_vcf()` 為 canonical API，支援 multi-allelic records。 |
| Pairwise/permutation 統計錯誤 | FIXED | 修正 summary、group defaults 與 permutation test 行為。 |
| Clustering/stacking 資料對齊 | FIXED | folds 依 sample identity 對齊；分類流程支援 stratification 與明確 features。 |
| OncoPlot 固定 GridSpec 與全域 pyplot side effect | FIXED | track-based declarative layout；`render(fig=...)` 不關閉全域 figures。 |
| 發版 metadata 與 upload 安全性 | FIXED | git tag 為版本單一來源、SPDX metadata、tag-gated publish。 |

## 部分完成

| 項目 | 狀態 | 剩餘工作 |
|---|---|---|
| PivotTable 拆分 | PARTIAL | IO、frequency、filter、sort、stats 已拆出；PCA/clustering 邏輯仍可再拆。 |
| Metadata alignment API | PARTIAL | 目前會 fail loud；仍缺 `relabel_features(mapper)` 這類同步改名 API。 |
| Schema validation | PARTIAL | MAF、PivotTable 核心與 TCGA inputs 已覆蓋；Cohort 跨表契約仍需專用 report。 |
| TCGA IO 雙軌 | PARTIAL | `io.tcga` builders 已成 canonical public API；舊 `tcga_readers.py` 尚未全面 thin-wrapper/deprecate。 |
| MAF variant-classification typo | PARTIAL | 已有正確 `valid_variant_classification` alias；舊拼字尚未發出 deprecation warning。 |
| Test coverage | PARTIAL | 已由 31% 提升至 65.19%；CNV、geneset 與部分整合流程仍偏低。 |

## 0.5.0 後續項目

以下是有效但不阻擋 0.5.0 的工作，優先順序由高到低：

1. 為 legacy `tcga_readers.py` 建立 thin wrappers 與 `DeprecationWarning`，消除兩套 TCGA IO 行為漂移。
2. 新增 `PivotTable.relabel_features()`，同步 data index 與 feature metadata；補 metadata alignment 回歸測試。
3. 補 MAF mutation feature metadata 與一致 key schema，提供 `format_feature_labels()`。
4. 補 Cohort 的 `sample_overlap()`、`missingness_report()` 與明確 sample alignment policy。
5. 將 library 內無條件 `print()` 收斂至 logging、warnings 或可注入 progress callback。
6. 未知 categorical 顏色不可靜默變白；應 warning 並在 legend 顯示 `Unknown`。
7. 縮減 `PivotTable.__getitem__` 與 pandas 直覺的差異，保留 `subset()` 作為 metadata-safe canonical API。
8. 將 coverage 提升至 85%，優先補 CNV、geneset、Cohort 與跨格式 round-trip。

## 明確不做

- 不把核心資料模型重寫成 AnnData。`to_anndata()` 是互通邊界，離散 Gene x Sample
  突變矩陣仍由 DataFrame-compatible `PivotTable` 擁有。
- 不自行實作完整跨 caller VCF normalization。production normalization 應交由
  `bcftools norm` 與專用 annotation tools；pymaftools 只負責已 harmonized rows 的讀取與共識整併。
- 不恢復 SQLite 作為主要儲存格式。其寬表 column limit 與 schema friction 不符合本專案資料形狀。
