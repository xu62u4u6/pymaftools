AI Agent 測試報告
=================

:日期: 2026-08-28
:分支: ``dev``
:整體結果: **部分通過（資產與安裝通過；部分 runtime 仍受環境限制）**

測試範圍
--------

本報告測試 PyMAFTools 內建的 AI 協作資產與建議工作流：

``scientific-user-tester`` → ``bioinformatics-researcher`` →
``bioinformatics-validator`` → ``biostatistics-reviewer`` →
``reproducibility-reviewer`` → ``scientific-manuscript-writer``。

這是角色責任的建議順序，不代表套件內建一個會自動串接的 orchestrator。測試分成
TOML 格式、安裝器、套件建置、文件與實際 runtime。只有真正載入指定角色並觀察到
其角色指令的測試，才算 runtime 測試通過；名稱相似的通用 child 不算。

內建 Agent
----------

``scientific-user-tester``
   只讀公開文件，從乾淨環境執行安裝、資料載入、表格與圖形輸出，記錄使用者會遇到
   的阻塞與文件缺口。

``bioinformatics-researcher``
   設計並執行探索性、可重現的生物資訊研究，再把結果交給獨立審查。

``bioinformatics-validator``
   獨立檢查研究設計、API 使用方式、統計方法、結果、生物解釋與限制。

``biostatistics-reviewer``
   攻擊 denominator、independence、confounding、selection、multiple testing、
   confidence interval、sparse cells 與 leakage。

``reproducibility-reviewer``
   獨立檢查其他人能否依文件、版本、checksum 與指令重跑流程，並產生實質一致的
   結果與圖形。

``scientific-manuscript-writer``
   依 claim ledger、可執行產物與獨立審查組裝論文，保留 VERIFIED、PARTIAL、
   PROPOSED、BLOCKED 與 NOT MEASURED 的證據界線。

自動化測試
----------

.. list-table::
   :header-rows: 1
   :widths: 32 18 50

   * - 測試項目
     - 結果
     - 證據
   * - Agent TOML 格式
     - 通過
     - 六個檔案都可由 ``tomllib`` 解析，並包含 ``name``、``description`` 與
       ``developer_instructions``。
   * - 安裝器單元測試
     - 通過
     - ``tests/test_ai_install.py`` 共 6 項測試通過；涵蓋六個 Agent 的 Codex
       檔案安裝、Claude Markdown 轉換與 dry-run。
   * - Ruff 靜態檢查
     - 通過
     - ``pymaftools`` 與 ``tests/test_ai_install.py`` 通過 Ruff。
   * - 完整專案測試
     - 通過
     - 304 項通過、7 項跳過、10 項未選取、48 warnings；coverage 68.52%。
   * - Wheel／sdist 套件內容
     - 通過
     - 網路建置的 wheel 與 sdist 通過 ``twine check``；wheel 內含六個 Agent
       定義檔。
   * - Codex 專案安裝
     - 通過
     - 六個 TOML 可由安裝器寫入隔離專案的 ``.codex/agents``。
   * - Claude 專案安裝
     - 通過
     - 六個 Markdown 可由同一份 TOML 寫入隔離專案的 ``.claude/agents``。

Claude 實際執行測試
-------------------

先前保存的 Claude Code 2.1.170 實測證據顯示，前三個 Agent 都能透過真正的
``--agent <name>`` 參數載入，且回答符合各自角色：

.. list-table::
   :header-rows: 1
   :widths: 32 22 46

   * - Agent
     - 執行結果
     - 角色證據
   * - ``bioinformatics-researcher``
     - 通過
     - 拒絕替自己的結論背書，要求交給兩個獨立審查角色。
   * - ``bioinformatics-validator``
     - 通過
     - 只使用 ``PASS``、``PASS WITH CONDITIONS`` 與 ``FAIL`` 三種審查狀態。
   * - ``reproducibility-reviewer``
     - 通過
     - 拒絕審查生物學解釋，只處理可重現性。

本輪新增的 ``scientific-user-tester``、``biostatistics-reviewer`` 與
``scientific-manuscript-writer`` 已完成格式、安裝與內容契約測試，但尚未取得新的
外部模型 runtime 證據。當前 shell 的 Claude Code 2.1.233 回報 organization
subscription access 被停用（需 Anthropic API key 或管理員開通），因此不能把
runtime 未執行寫成通過。

Claude 角色行為測試（既有證據）
--------------------------------

測試沒有把專案檔案傳給 Claude，而是使用不含私人 repository 內容的合成案例。

研究員
~~~~~~

結果：**修正後通過**。研究員針對六個合成 MAF 樣本產生探索性研究規格、資料來源
欄位、過濾規則、機器可讀輸出、研究限制與交接清單，並標示資料為合成資料，不得用
於臨床判斷。第一次測試發現 cosine distance 與 Ward linkage 不相容；角色定義已加
入方法相容性防呆，回歸測試改回 ``Euclidean distance + Ward linkage``。

生物資訊驗證員
~~~~~~~~~~~~~~

結果：**通過**。驗證員拒絕沒有研究設計支持的生存因果結論，指出缺少 cohort、參考
基因組、統計檢定、effect size、confidence interval 與多重檢定校正，且在沒有資料
時拒絕假裝重新計算，最後回傳 ``FAIL``。

可重現性審查員
~~~~~~~~~~~~~~

結果：**通過**。審查員指出刻意不完整的流程缺少輸入來源與 checksum、相依版本、
random seed、完整指令、參考基因組、預期輸出與比較容許誤差，最後回傳 ``FAIL``。

Codex 實際執行測試
------------------

測試環境：Codex CLI 0.149.1。

結果：**未通過——目前 runtime 無法選擇自訂角色**。先前已測試將 Agent TOML 放在
``.codex/agents``、隔離 Git repository、multi-agent 模式、stable
``multi_agent_v2`` 與 ``[agents.<role>]`` 註冊；Codex 可以建立通用 child thread，
但 child session metadata 的 ``agent_role`` 仍為空，且沒有載入套件提供的
``developer_instructions``。目前 ``spawn_agent`` 介面沒有 ``role`` 或
``agent_type`` 參數，因此不能把通用 child 宣稱為自訂 Agent 測試通過。

其他已確認限制
--------------

* Claude 的新三個角色目前只有格式、安裝與內容契約證據；外部模型權限恢復後，必須
  以不含私人資料的合成案例補做 runtime 行為測試。
* Claude 安全層不允許把 PyMAFTools repository 內容傳給外部模型，因此既有行為測試
  使用合成案例。
* PyMAFTools 現有的 ``compare_cohorts()`` 會回傳 odds ratio、p-value 與 FDR，
  但沒有 confidence interval。
* Codex 的自訂角色選擇是 runtime 相容性缺口，不是套件 Agent 定義格式錯誤。
* 本輪 Sphinx ``-W --keep-going`` 網路建置成功；離線建置仍會因 intersphinx
  inventory 的 DNS 解析而失敗，這是環境限制而非文件語法錯誤。

完整通過條件
------------

必須滿足以下條件，整套 Agent 流程才算完全通過：

#. 完成新增三個 Agent 的 Claude runtime 行為測試。
#. Codex 提供自訂角色選擇參數，而且 child session 記錄正確角色與套件內
   ``developer_instructions``。
#. 研究員、驗證員、統計審查員與可重現性審查員在後續回歸測試中持續依序交接，且
   每個結論都有可追溯證據。

官方格式來源
------------

目前 OpenAI Codex 手冊規定專案 Agent 應放在 ``.codex/agents``，並必須包含
``name``、``description`` 與 ``developer_instructions``：
`OpenAI Codex subagents 文件
<https://learn.chatgpt.com/docs/agent-configuration/subagents>`_。
