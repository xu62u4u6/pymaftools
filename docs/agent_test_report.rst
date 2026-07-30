AI Agent 測試報告
=================

:日期: 2026-07-30
:分支: ``dev``
:整體結果: **部分通過**

測試範圍
--------

本報告測試 PyMAFTools 內建的 AI 協作流程：

``bioinformatics-researcher`` → ``bioinformatics-validator`` →
``reproducibility-reviewer``。

測試分成三個層次：檔案格式、Agent 實際載入，以及角色行為。只有任務名稱相同的
通用子 Agent，不算自訂角色測試通過。

內建 Agent
----------

``bioinformatics-researcher``
   設計並執行探索性、可重現的生物資訊研究，再把結果交給獨立審查。

``bioinformatics-validator``
   獨立檢查研究設計、API 使用方式、統計方法、結果、生物解釋與限制。

``reproducibility-reviewer``
   獨立檢查其他人能否依文件重跑流程，並產生實質一致的結果。

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
     - 三個檔案都包含 ``name``、``description`` 與
       ``developer_instructions``
   * - 安裝器單元測試
     - 通過
     - ``tests/test_ai_install.py`` 共 6 項測試通過
   * - Ruff 靜態檢查
     - 通過
     - 安裝器與測試程式皆通過
   * - 完整專案測試
     - 通過
     - 264 項通過、7 項跳過、10 項未選取
   * - Wheel 套件內容
     - 通過
     - 離線建置的 wheel 包含三個 Agent 定義檔
   * - Codex 專案安裝
     - 通過
     - 三個 TOML 都成功安裝到 ``.codex/agents``
   * - Claude 專案安裝
     - 通過
     - 三個 Markdown 都成功安裝到 ``.claude/agents``

Claude 實際執行測試
-------------------

測試環境：Claude Code 2.1.170。

三個 Agent 都透過真正的 ``--agent <name>`` 參數載入。指令成功結束，而且每個
Agent 都依自己的角色指令回答：

.. list-table::
   :header-rows: 1
   :widths: 32 22 46

   * - Agent
     - 執行結果
     - 角色證據
   * - ``bioinformatics-researcher``
     - 通過
     - 拒絕替自己的結論背書，並要求交給兩個獨立審查角色
   * - ``bioinformatics-validator``
     - 通過
     - 只使用 ``PASS``、``PASS WITH CONDITIONS`` 與 ``FAIL`` 三種審查狀態
   * - ``reproducibility-reviewer``
     - 通過
     - 拒絕審查生物學解釋，只處理可重現性

Claude 角色行為測試
-------------------

測試沒有把專案檔案傳給 Claude，而是使用不含私人 repository 內容的合成案例。

研究員
~~~~~~

結果：**修正後通過**

研究員針對六個合成 MAF 樣本，產生探索性研究規格、資料來源欄位、過濾規則、
機器可讀輸出、研究限制與交接清單。它也清楚標示資料為合成資料，不得用於臨床
判斷。

第一次測試發現一個實質方法錯誤：研究員把 cosine distance 與 Ward linkage
搭配使用。Ward linkage 假設資料位於 Euclidean 空間，因此這個組合無效。

研究員定義已加入方法相容性防呆，明確禁止 Ward linkage 搭配 cosine distance。
修正後的 Claude 回歸測試回傳 ``Euclidean distance + Ward linkage``，並正確
說明兩者在數學上相容。

生物資訊驗證員
~~~~~~~~~~~~~~

結果：**通過**

驗證員收到一份刻意包含錯誤的六樣本研究報告後：

* 拒絕沒有研究設計支持的生存因果結論；
* 指出缺少 cohort 定義、參考基因組、統計檢定、effect size、confidence
  interval 與多重檢定校正；
* 在沒有資料時拒絕假裝重新計算；以及
* 最後正確回傳 ``FAIL``。

可重現性審查員
~~~~~~~~~~~~~~

結果：**通過**

審查員收到刻意不完整的 ``analysis.py data.maf`` 流程後，正確指出缺少：

* 輸入資料來源與 checksum；
* 相依套件版本；
* random seed；
* 完整執行指令；
* 參考基因組版本；
* 預期輸出；以及
* 結果比較容許誤差。

最後正確回傳 ``FAIL``。

Codex 實際執行測試
------------------

測試環境：Codex CLI 0.145.0。

結果：**失敗——目前 runtime 無法選擇自訂角色**

已測試以下方式：

#. 將獨立 Agent TOML 放在 ``.codex/agents``。
#. 在真正的隔離 Git repository 中執行。
#. 使用預設 multi-agent 模式。
#. 啟用 stable 的 ``multi_agent_v2``。
#. 在 ``.codex/config.toml`` 明確加入 ``[agents.<role>]`` 註冊。

Codex 可以建立通用 child thread，但 child session metadata 顯示
``agent_role: null``，也沒有載入套件提供的 ``developer_instructions``。
目前的 ``spawn_agent`` 介面沒有 ``role`` 或 ``agent_type`` 參數；明確註冊角色後
仍然沒有增加這些參數。

另外也測試了尚在開發中的 ``use_agent_identity``。它會嘗試註冊遠端 Agent
identity，但仍無法選擇專案內的自訂角色，因此不是可用的解法。

這是 runtime 相容性缺口：套件中的 TOML 符合目前官方文件的自訂 Agent 格式，
但這個 Codex CLI 版本無法透過現有的 spawn 介面選擇角色。名稱相似的通用 child
不能宣稱為自訂 Agent 測試通過。

其他已確認限制
--------------

* Claude 已成功載入三個角色，但安全層不允許把 PyMAFTools repository 內容傳給
  外部模型，所以改用合成案例測試角色行為。
* PyMAFTools 現有的 ``compare_cohorts()`` 會回傳 odds ratio、p-value 與 FDR，
  但沒有 confidence interval。
* 完整 Sphinx ``-W`` 建置仍有既有的 ``api/plot`` 未知 ``mpltype`` role，以及
  離線 intersphinx 警告。Agent 報告本身可以正常解析。

完整通過條件
------------

必須滿足以下條件，整套流程才算完全通過：

#. Codex 提供自訂角色選擇參數，而且 child session 記錄非空的正確角色與套件內
   ``developer_instructions``。
#. 研究員在後續回歸測試中持續選擇統計上相容的 distance/linkage 組合。
#. 在政策允許的情況下，完成一次真實研究，並把研究結果依序交給兩個獨立審查
   Agent。

官方格式來源
------------

目前 OpenAI Codex 手冊規定專案 Agent 應放在 ``.codex/agents``，並必須包含
``name``、``description`` 與 ``developer_instructions``：
`OpenAI Codex subagents 文件
<https://learn.chatgpt.com/docs/agent-configuration/subagents>`_。
