# Askingdoc — 项目宪法

> 本文件是给接手这个项目的 AI 看的规则文档，写的是**不随进度变化的规则**。
> 当前进度、已完成模块、待办事项在 [PROGRESS.md](PROGRESS.md)；
> 技术决策的完整"问题→方案→代价"论述在 [docs/DECISIONS.md](docs/DECISIONS.md)；
> 每一步详细的"①原理②手动任务③提示词"在
> [Askingdoc_dev_playbook_v2.md](Askingdoc_dev_playbook_v2.md)（原始规格，仍是权威原文）。
> 三份文档职责不重叠：这份讲"不能做什么"，PROGRESS 讲"做到哪了"，DECISIONS 讲"为什么这么做"。

---

## 1. 项目定位

Askingdoc 是一个长文档 RAG（检索增强生成）问答系统，个人 portfolio 项目，一周工期（9 个 STEP）。

面向数百页带文本层的技术文档（当前用例：一本 576 页的相机说明书），目标是让用户提问后，
系统从文档里检索相关片段、生成**带页码引用**的答案，并能对文档未覆盖的问题正确拒答。

**这个项目的核心交付物是四组消融实验数据（baseline/hybrid/rerank/full 各自的
Recall@5/Recall@20/MRR/Abstention Rate），不是功能数量。** 每加一个检索模块
（BM25、重排、small-to-big）都必须能用数字证明它有没有用——这是项目存在的意义，
也是评价"这一步做完了没有"的标准：光是代码跑通不算完成，必须有对应的评测数字。

---

## 2. 技术栈与选型理由

| 技术 | 用途 | 为什么选它 |
|---|---|---|
| Python 3.12 | 语言 | playbook 原定 3.11，环境搭建时已升级到 3.12，向后兼容无冲突 |
| PyMuPDF (pymupdf) | PDF 解析 | `get_text("blocks")` 模式自带版面块划分和坐标，不用自己实现版面分析 |
| LlamaIndex（仅 `SentenceSplitter`） | child 层切分 | 句子边界切分是成熟问题，不重复造轮子；但检索/融合/重排/生成全部手写，见【铁律】 |
| ChromaDB | 向量存储 | 本地持久化、无需额外部署数据库服务，`PersistentClient` 直接落盘 |
| rank_bm25 + jieba | 关键词检索 | 轻量、纯 Python、无需额外服务；jieba 处理中文分词，见【铁律】的 tokenize 约束 |
| BAAI/bge-m3（通过硅基流动 API） | embedding | 中文效果好、1024 维、有稳定 OpenAI 兼容 API，不用本地跑 embedding 模型 |
| BAAI/bge-reranker-v2-m3（本地） | 精排 | Cross-Encoder 重排必须读取候选原文，用 API 调用每次都要传全文本，本地跑更省成本也更快 |
| DeepSeek（deepseek-v4-flash） | 生成 | 中文能力强、成本低，适合个人项目预算 |
| FastAPI | STEP7 的 API 服务 | 自带 `/docs` 交互页面，面试展示比贴代码直观（尚未实现，见 PROGRESS.md） |

---

## 3.【铁律】—— 不可违反，改动前必须重新确认

> **这一节是本文件最重要的部分。任何 STEP 的实现都不能违反下面任何一条。**

### 🔒 `eval/harness.py` 和 `eval/golden_qa.jsonl` 已锁定
STEP3 完成后即锁死。**任何改动都会让已有的四组消融数据失去可比性**——评测代码或
评测题一变，之前跑出的 Recall@5/MRR 数字就不再是"同一把尺子"量出来的，STEP4/5
已经用这把尺子测出的数据全部作废。**禁止修改这两个文件，包括"看起来只是小修小补"的改动。**
后续任何 STEP 的提示词都必须带"严禁修改 eval/"。

### 🔒 检索链路必须手写，不用框架高层抽象（LlamaIndex 的 QueryEngine/Chain 等）
原因：
1. 需要拿到检索的**中间结果**（每个候选的 cid/page_no/score）直接喂给 `eval/harness.py`
   算 Recall@5/@20/MRR——高层抽象把这些中间状态封装掉了，测不到。
2. 四个消融开关（`use_bm25`/`use_rerank`/`use_small_to_big`）需要能**单变量控制**，
   高层框架的 pipeline 通常不支持这种细粒度开关组合。

LlamaIndex 唯一被允许使用的地方是 `src/ingest/chunker.py` 里的 `SentenceSplitter`
（纯文本切分，不涉及检索逻辑）。

### 🔒 `page_no` 必须是 PDF 原始页码，不能重新编号
`src/ingest/parser.py` 解析时，即使 dev 模式只截取前 150 页，页码也必须是
PyMuPDF 原始页码 + 1（PyMuPDF 从 0 开始），**绝不能因为只处理了 150 页就从 1 重新编号**。
一旦重新编号，STEP6 切到全量 576 页时，`eval/golden_qa.jsonl` 里所有 `gold_pages`
全部错位，且**无法事后修复**（原始信息已经在解析阶段丢失）。

### 🔒 dev 和 full 必须共用同一份代码
仅由 `config.dev.enabled` 这一个布尔值切换，所有路径和 ChromaDB 集合名通过
`{suffix}` 占位符自动隔离（`_dev` 或空字符串）。**禁止出现"dev 版函数"和
"full 版函数"两套逻辑**——那样 STEP6 切换时会有大量不一致的风险。

### 🔒 `search()` 必须返回结构化 `Hit` 列表，不能返回拼好的字符串
`Hit` dataclass 字段：`cid, text, page_no, score, parent_id, rerank_score`
（最后一个字段仅重排后有值，其余情况为 `None`）。原因：`eval/harness.py` 需要
机械地读取 `hit.page_no` 去和 `gold_pages` 比对——如果返回的是"第87页提到..."
这种拼好的字符串，评测代码就要用正则去抠页码，脆弱且容易出错。

### 🔒 tokenize 函数只有一份，建索引和查询必须调同一个
`src/ingest/tokenizer.py` 的 `tokenize()` 是全项目唯一的分词实现。
`src/ingest/indexer.py`（建 BM25 索引）和 `src/retrieval/search.py` 的
`bm25_search()`（查询时分词）**必须 import 同一个函数**，不允许任何一处
自己重新写分词逻辑。不一致的后果：BM25 会**静默失效**——不报错，只是
检索结果普遍不准，因为查询词和语料词的字符串对不上，这个 bug 极难排查。

---

## 4. 架构说明

```
离线 ingest 流水线：
  PDF (带页码的文本层)
    │  src/ingest/parser.py — PyMuPDF get_text("blocks")
    ▼
  blocks (每条带原始 page_no，噪声过滤后)
    │  src/ingest/chunker.py — 顺序累积 blocks 成 parent (~1500 token)
    │                          再用 LlamaIndex SentenceSplitter 切 child (~300 token)
    ▼
  parent (完整上下文，供生成用) ──┐  child (精确聚焦，供检索用)
                                  │       │
                                  │       ▼  src/ingest/indexer.py
                                  │   ChromaDB(向量) + bm25.pkl(词，仅存分词结果)
                                  │
                                  └─── (STEP5 才用到，small-to-big 扩展时换回 parent)

在线 query 流水线（src/retrieval/search.py 的 search()）：
  query
    │  vector_search() — 向量召回 top-20（无条件执行）
    ▼
  [if use_bm25]   bm25_search() + fusion.py 的 rrf_fuse() 融合 BM25 top-20
    ▼
  [if use_rerank] reranker.py 的 rerank() 用 Cross-Encoder 精排，截到 final_top_k(5)
    ▼
  [if use_small_to_big]  child → 对应 parent 扩展，去重补足（STEP5，尚未实现）
    ▼
  src/generation/answerer.py 的 answer() — 拼 prompt，调 LLM，生成带页码引用的回答
```

三个方括号是消融开关，均由 `config.retrieval.*` 的布尔值控制，四种组合对应
`eval/configs.py` 的 `baseline`/`hybrid`/`rerank`/`full` 四组预设。

---

## 5. 双数据集机制

`config.dev.enabled` 决定 `{suffix}` 是 `"_dev"` 还是 `""`，`src/config.py` 的
`load_config()` 在读取 `config.yaml` 后立即解析这个占位符，所有 `paths.*` 和
`index.collection_name` 据此自动隔离，物理上不共享同一份存储/集合。

| | dev | full |
|---|---|---|
| 页数 | 前 150 页 | 全量 576 页 |
| 评测题数 | 12 条（`in_dev_range=true`） | 20 条（STEP6 补到 20，全部跑） |
| 结果文件 | `eval/results/dev_*.json` | `eval/results/full_*.json`（尚不存在） |
| chroma 集合 | `askingdoc_dev` | `askingdoc` |

`load_config(overrides: dict)` 支持点号路径覆盖任意配置项，同时服务于两个场景：
消融实验（`eval/configs.py` 切 `retrieval.*` 开关）和 dev/full 切换
（STEP6 切 `dev.enabled`）——不是两套机制，是同一个函数的两种用法。

---

## 6. 代码规范

- 注释和 docstring 用英文，要**具体说明逻辑/原因**，不要复述函数名
- 路径一律用 `pathlib`，不用字符串拼接
- 关键中间结果要打印出来（统计量、样本抽查），便于调试和人工核验
- 每个模块级单例（chroma client、bm25 index、reranker 等）都要提供 `reset_*()`
  函数——不是可选项，是双模式切换和长进程复用场景下的硬需求

---

## 7. 协作方式

**每次任务明确改动范围，不越界修改其他文件；单个问题卡住超 60 分钟就降级实现
或标记为已知限制，不要死磕。**

补充这个项目实际验证有效、建议延续的工作习惯（不是新规则，是过去开发中形成的模式）：

- **每个 STEP/Slice 先讲原理再写代码**：用户是 RAG 初学者，目的除了交付作品也是
  通过实现学懂原理。实现前先用大白话讲这一步在 RAG 全局流程里的位置、要解决什么矛盾，
  实现后逐条回答提示词末尾的"输出后请解释"问题（挑关键行讲"为什么这样写"，不是复述代码）。
- **评测题必须人工手写，不能 AI 代写**：`eval/golden_qa.jsonl` 是全项目唯一不能交给 AI
  生成的内容——AI 从 chunk 反向生成的问题会和 chunk 高度同源，检索必然命中，Recall 虚高，
  测不出真实能力。这个项目里实际发生过一次例外（用户手写8条、AI补4条），当时已明确告知
  AI补充部分存在"同源"风险，用户知情后接受——但默认应坚持全部人工手写。
- **实现完要用真实数据验证，不能只保证语法正确**：这个项目里多次出现"代码逻辑对但没用
  真实数据测过就想当然"导致的问题（例如 STEP4 的 `recall_top_k` 截断 bug，直到用
  `explain_rerank` 实测才发现候选数是 38 条而非预期的 20 条）。新功能实现后应尽量跑一次
  真实调用（哪怕只是一个样例 query），而不是只做 import/语法检查。
- **付费 API 调用（embedding/LLM/大规模索引）默认不擅自跑全量**，尤其涉及用户自己的
  API 余额时，先用小范围验证（如 `--limit 20`），等用户明确说"可以，去跑"或类似授权后
  再执行完整的、有实际花费的操作。
- **写代码时如果发现文档描述和实际代码/数据不一致，要如实指出，不要沉默地"顺便"改掉**——
  这个项目里 CLAUDE.md、`docs/capability_log.md` 的进度记录多次因为实现推进而过时，
  发现后应该主动提出更新，而不是留着不管。
