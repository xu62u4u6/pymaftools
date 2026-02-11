# pymaftools Optimization Roadmap

> Last updated: 2026-02-11

## Phase 1: Foundation (Week 1-2)

### 1.1 Fix Critical Code Quality Issues

- [ ] Replace bare `except:` with specific exceptions in `MAF.py:168,183`
  > **Why:** 裸 `except` 會捕獲所有異常（包括 `KeyboardInterrupt`、`SystemExit`），導致嚴重錯誤被靜默吞掉，極難除錯。

  ```python
  # Bad (MAF.py:168)
  try:
      return int(pos)
  except:
      return None

  # Good
  try:
      return int(pos)
  except (ValueError, TypeError):
      return None
  ```

- [ ] Replace wildcard imports in `__init__.py` with explicit imports
  > **Why:** `from .core.Clustering import *` 會污染命名空間，造成名稱衝突且無法從 import 語句看出實際導入了什麼，降低可讀性。

  ```python
  # Bad (__init__.py:12)
  from .core.Clustering import *

  # Good
  from .core.Clustering import Clustering, ClusterResult
  ```

- [ ] Resolve TODO comments in `PivotTable.py:247,284`
  > **Why:** TODO 標記表示未完成的重構（`False` vs `"WT"` 不一致），長期留存會導致資料表示不統一，引發下游分析錯誤。

  ```python
  # Bad (PivotTable.py:247) — 用 False 和 "WT" 混合表示野生型
  table = self.copy().rename_index_and_columns()
  # TODO: replace False with "WT" in all files
  table = table.replace(False, "WT")

  # Good — 定義常數統一表示，移除 TODO
  WILD_TYPE = "WT"
  # 內部一律使用 WILD_TYPE，不再混用 False/"WT"
  table = self.copy().rename_index_and_columns()
  table = table.replace(False, WILD_TYPE)
  ```

### 1.2 Dependency Version Constraints

- [ ] Add upper bounds to all dependencies in `pyproject.toml`
  > **Why:** `numpy` 完全無版本約束、`pandas>2.0` 無上界，當依賴發布破壞性更新（如 numpy 2.0 移除大量 API）時，用戶安裝後會直接報錯。

  ```toml
  # Bad (pyproject.toml:17-31)
  dependencies = [
    "pandas>2.0",
    "numpy",
    "networkx",
    "matplotlib",
    # ...
  ]

  # Good
  dependencies = [
    "pandas>=2.0,<3.0",
    "numpy>=1.24,<3.0",
    "networkx>=3.0,<4.0",
    "matplotlib>=3.7,<4.0",
    # ...
  ]
  ```

- [ ] Consider making `beautifulsoup4`, `statsmodels` optional dependencies
  > **Why:** `beautifulsoup4` 僅在 `geneset.py` 一處使用，`statsmodels` 安裝體積大且編譯慢，設為 optional 可大幅減少基礎安裝時間。

  ```toml
  # Bad — 所有用戶都必須安裝
  dependencies = ["beautifulsoup4", "statsmodels"]

  # Good — 按需安裝
  dependencies = [...]  # 僅核心依賴

  [project.optional-dependencies]
  web = ["beautifulsoup4", "requests"]
  stats = ["statsmodels"]
  all = ["beautifulsoup4", "requests", "statsmodels"]
  ```

### 1.3 CI/CD Improvements

- [ ] Add `ruff` linting to CI workflow
  > **Why:** 目前 CI 僅跑測試，無靜態分析，代碼風格不一致和潛在 bug 只能靠人工 review 發現。

  ```yaml
  # Good — 在 tests.yml 中新增 lint job
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install ruff
      - run: ruff check .
      - run: ruff format --check .
  ```

- [ ] Add `mypy` type checking to CI workflow
  > **Why:** 配合 Phase 2 的 type hints，可在 CI 自動捕獲型別錯誤，防止 `None` 傳入不支援的函數等常見問題。
- [ ] Add test coverage reporting (pytest-cov)
  > **Why:** 目前無法量化測試覆蓋率，無從得知哪些路徑完全沒被測試保護。
- [ ] Add coverage threshold enforcement (e.g. fail below 40%)
  > **Why:** 設定最低門檻可防止新 PR 進一步降低覆蓋率，建立持續改善的機制。

  ```yaml
  # Good — coverage 配置範例
  - run: pytest --cov=pymaftools --cov-fail-under=40 --cov-report=xml
  ```

---

## Phase 2: Type Safety & Testing (Week 3-6)

### 2.1 Add Type Hints

- [ ] Core modules: `MAF.py`, `PivotTable.py`, `Cohort.py`
- [ ] Plot modules: `OncoPlot.py`, `LollipopPlot.py`
- [ ] Model modules: `StackingModel.py`
- [ ] Utils modules: `geneset.py`, `reduction.py`, `geneinfo.py`

> **Why:** 目前約 95% 的函數缺少 type hints，IDE 無法提供自動補全和錯誤提示，使用者必須查看源碼才能了解參數型別，開發效率低且容易傳錯參數。

```python
# Bad (PivotTable.py:153) — 無型別標註
def _copy_metadata(self, source):
    """Safely copy metadata attributes from another object."""
    ...

# Bad (geneset.py:17) — 參數和返回值皆無標註
def fetch_msigdb_geneset(geneset_name, species="human"):
    ...

# Good
def _copy_metadata(self, source: "PivotTable") -> None:
    """Safely copy metadata attributes from another object."""
    ...

def fetch_msigdb_geneset(geneset_name: str, species: str = "human") -> list[str]:
    ...
```

### 2.2 Increase Test Coverage (target: 60%+)

- [ ] Add tests for `MAF.py` (currently 0%)
- [ ] Add tests for `CopyNumberVariationTable.py` (currently 0%)
- [ ] Add tests for `model/StackingModel.py` (currently 0%)
- [ ] Add tests for `utils/geneset.py` (currently 0%)
- [ ] Add edge case tests: empty DataFrames, invalid inputs, unicode gene names
- [ ] Add integration tests for Cohort + PivotTable workflows

> **Why:** 當前測試覆蓋率僅 ~12%，MAF（核心入口）、ML 模型、工具函數完全沒有測試。任何重構都可能悄悄破壞功能而無法察覺，嚴重阻礙後續 Phase 3 的架構改造。

```python
# 目前缺少的測試範例

# tests/core/test_maf.py — 目前不存在
class TestMAF:
    def test_read_maf_basic(self, tmp_path):
        """測試基本 MAF 讀取"""
        maf = MAF.read_maf("test_data/sample.maf")
        assert len(maf) > 0
        assert "Hugo_Symbol" in maf.columns

    def test_read_maf_empty_file(self, tmp_path):
        """測試空檔案的邊界情況"""
        empty_file = tmp_path / "empty.maf"
        empty_file.write_text("")
        with pytest.raises(ValueError):
            MAF.read_maf(str(empty_file))

    def test_parse_position_invalid(self):
        """測試非法位置值"""
        assert MAF._parse_position("abc") is None
        assert MAF._parse_position("123") == 123
```

---

## Phase 3: Architecture & Performance (Week 7-10)

### 3.1 Refactor PivotTable (1,692 lines -> smaller modules)

- [ ] Extract statistical methods to `core/statistics.py`
- [ ] Extract PCA/clustering logic to `core/analysis.py`
- [ ] Extract persistence (SQLite/HDF5) to `core/persistence.py`
- [ ] Keep data operations in `PivotTable.py`

> **Why:** `PivotTable.py` 承擔了資料存儲、統計檢定、PCA/聚類、持久化、繪圖等過多職責（God Object），單一修改容易牽連其他功能，且 1,692 行的文件難以閱讀和維護。

```python
# Bad — 所有邏輯都在 PivotTable 類中 (1,692 lines)
class PivotTable(pd.DataFrame):
    def calculate_feature_frequency(self): ...
    def chi_squared_test(self): ...
    def fisher_exact_test(self): ...
    def pca(self): ...
    def clustering(self): ...
    def to_sqlite(self): ...
    def to_hdf5(self): ...
    def read_hdf5(cls): ...
    # ... 數十個方法

# Good — 拆分為獨立模組
# core/statistics.py
def chi_squared_test(pivot_table: PivotTable, ...) -> pd.DataFrame: ...
def fisher_exact_test(pivot_table: PivotTable, ...) -> pd.DataFrame: ...

# core/analysis.py
def run_pca(pivot_table: PivotTable, ...) -> PCAResult: ...
def run_clustering(pivot_table: PivotTable, ...) -> ClusterResult: ...

# core/persistence.py
def to_hdf5(pivot_table: PivotTable, path: str) -> None: ...
def read_hdf5(path: str) -> PivotTable: ...

# core/PivotTable.py — 只保留資料操作
class PivotTable(pd.DataFrame):
    def calculate_feature_frequency(self): ...
    def subset(self): ...
    def merge(self): ...
```

### 3.2 Performance Optimization

- [ ] Add memoization/caching for `calculate_feature_frequency()`
  > **Why:** 該方法被多處重複調用（繪圖、排序、篩選），每次都重新計算整個 DataFrame 的頻率，在大型資料集上造成明顯延遲。

  ```python
  # Bad — 每次調用都重新計算
  class PivotTable:
      def calculate_feature_frequency(self) -> pd.Series:
          binary = (self != False).astype(bool)
          return binary.sum(axis=1) / binary.shape[1]

      def plot(self):
          freq = self.calculate_feature_frequency()  # 計算第 1 次
          ...
      def sort_by_freq(self):
          freq = self.calculate_feature_frequency()  # 計算第 2 次（重複）
          ...

  # Good — 使用快取，資料變更時失效
  class PivotTable:
      _freq_cache: Optional[pd.Series] = None

      def calculate_feature_frequency(self) -> pd.Series:
          if self._freq_cache is None:
              binary = (self != False).astype(bool)
              self._freq_cache = binary.sum(axis=1) / binary.shape[1]
          return self._freq_cache

      def __setitem__(self, key, value):
          self._freq_cache = None  # 資料變更時清除快取
          super().__setitem__(key, value)
  ```

- [ ] Reduce unnecessary `.copy()` calls across codebase
  > **Why:** 多處在只需讀取的場景下仍做完整 DataFrame 複製，浪費記憶體，對大型 MAF 資料（數萬筆突變）影響顯著。

  ```python
  # Bad (reduction.py:136) — 只是為了加一欄就複製整個 DataFrame
  df = df.copy()
  df["abs_weight_mean"] = df.abs().mean(axis=1)

  # Good — 使用 assign 返回新 DataFrame（更 pandas 慣用且語義清晰）
  df = df.assign(abs_weight_mean=lambda x: x.abs().mean(axis=1))
  ```

- [ ] Add `timeout` parameter to `requests.get()` in `geneset.py`
  > **Why:** 無 timeout 的 HTTP 請求在網路異常時會無限期掛起，導致整個程式凍結，用戶只能強制終止。

  ```python
  # Bad (geneset.py:20)
  res = requests.get(url)

  # Good
  res = requests.get(url, timeout=30)
  ```

- [ ] Consider lazy loading for large data files
  > **Why:** `protein_domains.csv` 在 import 時即載入 24MB 資料，即使用戶不使用 LollipopPlot 功能也要付出啟動成本。

  ```python
  # Bad (MAF.py:204-208) — import 時立即載入 24MB
  script_dir = os.path.dirname(os.path.abspath(__file__))
  protein_domains_path = os.path.join(script_dir, "../data/protein_domains.csv")
  protein_domains = pd.read_csv(protein_domains_path, index_col=0, low_memory=False)

  # Good — 首次使用時才載入
  _protein_domains: Optional[pd.DataFrame] = None

  def get_protein_domains() -> pd.DataFrame:
      global _protein_domains
      if _protein_domains is None:
          path = os.path.join(os.path.dirname(__file__), "../data/protein_domains.csv")
          _protein_domains = pd.read_csv(path, index_col=0, low_memory=False)
      return _protein_domains
  ```

### 3.3 Move Large Data Files

- [ ] Move `protein_domains.csv` (24MB) out of git repository
- [ ] Implement download-on-first-use with local caching
- [ ] Or publish as a separate data package

> **Why:** 24MB 的 CSV 文件讓 git clone 變慢、package 體積膨脹。每次 clone 都要下載完整歷史中的大文件，對 CI 和新開發者不友好。

```python
# Bad — 大檔案直接放在 repo 中
pymaftools/data/protein_domains.csv  # 24MB in git

# Good — 按需下載 + 本地快取
from platformdirs import user_cache_dir

CACHE_DIR = Path(user_cache_dir("pymaftools"))
DOMAINS_URL = "https://github.com/.../releases/download/v1/protein_domains.parquet"

def get_protein_domains() -> pd.DataFrame:
    cache_path = CACHE_DIR / "protein_domains.parquet"
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DOMAINS_URL, cache_path)
    return pd.read_parquet(cache_path)  # parquet 比 CSV 快 10x
```

---

## Phase 4: Documentation & DX (Week 11-14)

### 4.1 Documentation

- [ ] Set up Sphinx with autodoc
  > **Why:** 目前無 API 文檔網站，使用者只能讀源碼或 README 的有限範例來學習 API，阻礙項目被更多研究者採用。

- [ ] Standardize docstrings to NumPy style
  > **Why:** 目前混用 Google style 和 NumPy style，風格不統一導致 Sphinx 解析異常，也增加貢獻者的認知負擔。

  ```python
  # Bad — Google style (BasePlot.py:26)
  def add_legend(self, legend_name: str, color_dict: dict):
      """
      Add legend information to LegendManager

      Args:
          legend_name: Legend name, such as 'mutation', 'sex', 'subtype', etc.
          color_dict: Color mapping dictionary

      Returns:
          self: Support method chaining
      """

  # Good — 統一使用 NumPy style
  def add_legend(self, legend_name: str, color_dict: dict) -> "BasePlot":
      """
      Add legend information to LegendManager.

      Parameters
      ----------
      legend_name : str
          Legend name, such as 'mutation', 'sex', 'subtype', etc.
      color_dict : dict
          Color mapping dictionary, e.g. ``{'M': 'blue', 'F': 'red'}``.

      Returns
      -------
      BasePlot
          Self, to support method chaining.
      """
  ```

- [ ] Unify code comments to English
  > **Why:** `ColorManager.py` 等文件含中文註釋，對國際貢獻者造成閱讀障礙，不利於開源社區參與。

  ```python
  # Bad (ColorManager.py:144-167)
  unique_categories = pd.unique(data.values.ravel())  # 獲取唯一類別
  unique_categories = [cat for cat in unique_categories if pd.notna(cat)]  # 移除 NaN 值
  if custom_cmap:  # 如果有自定義映射，則覆蓋默認顏色
      for category, color in custom_cmap.items():
          ...

  # Good
  unique_categories = pd.unique(data.values.ravel())
  unique_categories = [cat for cat in unique_categories if pd.notna(cat)]
  if custom_cmap:  # Override defaults with custom color mapping
      for category, color in custom_cmap.items():
          ...
  ```

- [ ] Add architecture overview diagram
  > **Why:** 模組間的依賴關係（MAF -> PivotTable -> Cohort -> Plot）缺乏視覺化說明，新開發者難以快速理解項目結構。
- [ ] Deploy docs to GitHub Pages
  > **Why:** 讓 API 文檔可通過 URL 直接訪問，無需本地 build，降低使用門檻。

### 4.2 Developer Experience

- [ ] Add `CONTRIBUTING.md`
  > **Why:** 缺少貢獻指南會讓外部開發者不知道如何參與（代碼風格、PR 流程、測試要求），降低社區參與意願。
- [ ] Add `.pre-commit-config.yaml` (ruff, mypy)
  > **Why:** 在 commit 前自動檢查代碼品質，避免不合規的代碼進入 repo，減少 CI 上的來回修改。

  ```yaml
  # Good — .pre-commit-config.yaml
  repos:
    - repo: https://github.com/astral-sh/ruff-pre-commit
      rev: v0.9.0
      hooks:
        - id: ruff
          args: [--fix]
        - id: ruff-format
    - repo: https://github.com/pre-commit/mirrors-mypy
      rev: v1.14.0
      hooks:
        - id: mypy
          additional_dependencies: [pandas-stubs]
  ```

- [ ] Add automatic PyPI publishing workflow
  > **Why:** 目前發版需手動操作，容易遺漏步驟或版本號不一致，自動化可確保每次 release tag 都正確發佈到 PyPI。

  ```yaml
  # Good — .github/workflows/publish.yml
  on:
    release:
      types: [published]
  jobs:
    publish:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - run: pip install build twine
        - run: python -m build
        - run: twine upload dist/*
          env:
            TWINE_USERNAME: __token__
            TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
  ```

- [ ] Add changelog generation (e.g. `git-cliff`)
  > **Why:** 手動維護 changelog 容易遺漏，自動從 commit message 生成可確保完整性，也讓使用者快速了解版本間的變化。

---

## Phase 5: Feature Enhancements (Week 15+)

### 5.1 Export & Compatibility

- [ ] Add Parquet export support
  > **Why:** Parquet 是列式存儲格式，壓縮率高、讀取速度快，適合生物資訊中常見的寬表（大量基因列），相比 CSV 可節省 5-10x 空間。

  ```python
  # Good — 新增 Parquet 支援
  def to_parquet(self, path: str, compression: str = "zstd") -> None:
      self.to_parquet(path, compression=compression)
  ```

- [ ] Add Excel export support
  > **Why:** 許多臨床研究人員習慣使用 Excel 查看數據，缺少直接導出功能會增加他們的使用摩擦。
- [ ] Add compression options for exports
  > **Why:** MAF 資料檔案通常很大，提供 gzip/zstd 壓縮選項可大幅減少磁碟佔用和傳輸時間。

### 5.2 Advanced Analytics

- [ ] Add survival analysis support
  > **Why:** 生存分析（Kaplan-Meier、Cox regression）是癌症基因組學的核心分析之一，目前需要用戶自行整合外部工具。

  ```python
  # Good — 整合 lifelines 進行生存分析
  from lifelines import KaplanMeierFitter

  def kaplan_meier(self, time_col: str, event_col: str,
                   group_col: str | None = None) -> KaplanMeierResult:
      ...
  ```

- [ ] Add cross-validation utilities for ML models
  > **Why:** `StackingModel` 目前缺少標準化的交叉驗證流程，用戶容易產生過擬合而不自知。

  ```python
  # Bad — 用戶需自行實作交叉驗證
  model = StackingModel(...)
  model.fit(X_train, y_train)
  score = model.score(X_test, y_test)  # 可能過擬合而不知

  # Good — 內建交叉驗證
  model = StackingModel(...)
  cv_results = model.cross_validate(X, y, cv=5, scoring="roc_auc")
  print(cv_results.mean_score, cv_results.std_score)
  ```

- [ ] Add more statistical tests (ANOVA, Kruskal-Wallis)
  > **Why:** 目前僅支援 chi-squared 和 Fisher's exact test，無法滿足連續變量比較（如基因表達量在不同突變組之間的差異）的需求。

### 5.3 User Experience

- [ ] Add CLI tool for common operations
  > **Why:** 讓不熟悉 Python 的研究人員也能通過命令列快速執行常見分析（如生成 OncoPlot），降低使用門檻。

  ```bash
  # Good — CLI 使用範例
  pymaftools oncoplot --maf data/tcga.maf --genes TP53 KRAS EGFR -o oncoplot.png
  pymaftools summary --maf data/tcga.maf --output report.html
  ```

- [ ] Consistent progress bar usage (tqdm)
  > **Why:** 部分耗時操作有進度條、部分沒有，用戶在處理大型資料時無法判斷程式是否仍在運行。
- [ ] Add structured logging (replace print statements)
  > **Why:** `print` 語句無法控制輸出級別，用戶無法區分 debug 資訊和重要警告，也無法將日誌重定向到文件。

  ```python
  # Bad (PivotTable.py:253, FontManager.py:58)
  print(f"[PivotTable] saved to {db_path}")
  print(f"Font already registered: {target_name}")
  print("Warning: No valid paired samples found. Switching to unpaired test.")

  # Good
  import logging
  logger = logging.getLogger(__name__)

  logger.info("Saved to %s", db_path)
  logger.debug("Font already registered: %s", target_name)
  logger.warning("No valid paired samples found. Switching to unpaired test.")
  ```

- [ ] Consider interactive plots (plotly) as optional feature
  > **Why:** 靜態圖片無法 hover 查看具體數值，在 Jupyter Notebook 環境中互動式圖表能大幅提升數據探索效率。

---

## Metrics

| Metric | Current | Phase 2 Target | Phase 4 Target |
|--------|---------|----------------|----------------|
| Test Coverage | ~12% | 60% | 85% |
| Type Hints | ~5% | 70% | 95% |
| Largest File | 1,692 lines | < 800 lines | < 500 lines |
| CI Checks | tests only | tests + lint + types | + coverage + docs |
