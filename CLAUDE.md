# Askingdoc — 项目工作指南（供后续轮次使用）

> 本文件是根据 `Askingdoc_dev_playbook_v2.md`（下称"playbook"）提炼的工作契约。
> playbook 是每一步**提示词的权威原文**，遇到细节冲突以 playbook 为准；
> 本文件负责：项目全局约束 + 每次交互的协作方式 + 当前进度追踪。

---

## 1. 项目是什么

Askingdoc：长文档 RAG（检索增强生成）问答系统。个人项目，一周工期（9 个 STEP）。
输入是一本 300–600 页、带文本层的 PDF（相机说明书），目标是让用户提问后，系统从文档里
检索相关片段、生成带**页码引用**的答案，并能拒答文档未覆盖的问题。

**技术栈**：Python 3.12 / PyMuPDF / LlamaIndex（仅用于切分）/ ChromaDB / rank_bm25 /
embedding API / bge-reranker-v2-m3 / FastAPI

**用户身份**：项目所有者是 RAG 领域的初学者，做这个项目的目的除了交付作品，
也是**通过亲手实现来学懂 RAG 全套流程**（解析→切分→向量化→检索→融合→重排→生成→引用）。
这一点决定了下面第 4 节的协作方式，比代码规范更重要，不要跳过。

---

## 2. 设计原则（不可违反）

1. **检索链路手写，不用框架高层抽象**。LlamaIndex 只用来做 SentenceSplitter 切分，
   检索、融合、重排、生成全部自己写。这是为了能在面试里讲清楚每一步原理。
2. **四个消融开关（use_bm25 / use_rerank / use_small_to_big + 隐含的 baseline）
   全部由 `config.yaml` 控制，一份代码跑四种配置**。`src/retrieval/search.py` 里
   `search()` 函数按 STEP2 定的骨架（四个 if 分支）填充，不要为不同配置写不同函数。
3. **双数据集模式（dev / full）共用同一份代码**，仅由 `config.dev.enabled` 切换：
   - dev：前 150 页，独立 chroma 集合 `askingdoc_dev`，独立 `bm25_dev.pkl`，12→之后补到 20 条评测题
   - full：全量页数，正式集合 `askingdoc`，正式 `bm25.pkl`
   - 所有路径 / 集合名通过 `{suffix}` 占位符自动区分（`_dev` 或空字符串），
     **绝不允许出现"dev 版代码"和"正式版代码"两套逻辑**
4. **`page_no` 必须在 STEP1（PDF 解析）就保存，且必须是 PDF 原始页码**，
   即使 dev 模式只处理前 150 页也不能重新编号——否则引用溯源做不了，
   STEP6 切 full 时评测集 `gold_pages` 会全部错位。
5. **模型固定，不擅自更换**：凡涉及 embedding 的地方（`config.yaml` 的
   `models.embedding`、indexer 的向量化调用、search 的 query 向量化）一律用
   `BAAI/bge-m3`（向量维度 1024）；凡涉及 LLM 生成的地方（`config.yaml` 的
   `models.llm`、`answerer.py` 的生成调用）一律用 `deepseek-v4-flash`。
   写代码、写提示词示例、写文档举例时都用这两个名字，除非用户明确说要换模型。
6. **密钥只从 `.env` 拿，不允许硬编码或写进 `config.yaml`**：任何要用到
   API key / base_url 的代码（embedding 调用、LLM 调用），一律通过
   `src/config.py` 的 `load_config()` 取 `Config.embedding_api_key` /
   `embedding_base_url` / `llm_api_key` / `llm_base_url` 这四个字段，
   不要在新模块里自己重新 `load_dotenv()` 或读 `os.environ`，
   也不要把真实 key/url 写进代码、`config.yaml` 或对话里贴出来。
   `.env` 已在 `.gitignore` 里，`.env.example` 是给别人看的空模板。

---

## 3. 代码规范

- 注释和 docstring 用英文，要**具体说明逻辑/原因**，不要复述函数名
- 路径一律用 `pathlib`，不用字符串拼接
- 关键中间结果要打印出来（统计量、样本抽查），便于调试和人工核验
- 每次交付代码后，**逐行解释关键逻辑的工作原理**（不是整体概述，是挑关键行讲清楚"为什么这样写"）
- 每个 STEP 的提示词末尾都有【输出后请解释】几个具体问题，**必须逐条回答**，
  这是用户学习 RAG 原理的主要途径，不能省略或一笔带过

---

## 4. 每一步的协作方式（本项目最重要的约定）

用户是 RAG 初学者，希望**边做边学懂原理**，而不是拿到一堆能跑的代码就完事。
因此对 playbook 里的**每一个 STEP / 每一个 Slice**，交互都按两步走，不能合并、不能跳步：

**第一步：讲清楚"这步在干什么、怎么干的"**
- 用大白话讲这一步在 RAG 全局流程里的位置（解析？切分？索引？检索？融合？重排？生成？）
- 讲清楚要解决什么矛盾/问题（例如 small-to-big 解决"检索要小块、生成要大块"的矛盾）
- 可以直接引用 playbook 该 STEP 的"① 这步在干什么"内容，但要用自己的话讲透，
  不是原文照搬；必要时举例子（结合相机说明书这个具体场景）
- 讲清楚"② 你要手动做什么"里哪些事需要用户亲自动手（人工抽查、手写评测题等），
  并说明**为什么这一步不能交给 AI**（playbook 里通常写了原因，比如评测题让 AI 生成会同源导致 Recall 虚高）

**第二步：实现 playbook 该 STEP/Slice 的③提示词内容**
- 严格按提示词里的【改动范围】执行，不越界修改其他文件
- 严格遵守【严禁修改 xxx】类约束（尤其 STEP3 之后 `eval/harness.py` 和
  `eval/golden_qa.jsonl` 锁死，任何 STEP 都不能碰）
- 实现完，逐条回答提示词末尾【输出后请解释】的问题
- 给出这一步的验收方法（playbook 里"**验收**"部分写了怎么跑、该看到什么结果）

**不要一次把多个 STEP 的代码都生成出来**——即使看到 playbook 后面步骤的内容，
也只做用户当前明确要求的这一步，除非用户说"接下来几步一起做"。

**每个 STEP 前后建议 `git commit`**（playbook 全局纪律第1条）。commit message
用 playbook 里给的格式（如 `step0: skeleton with dev/full dual-dataset config`），
但**只在用户确认要提交时才执行 `git commit`**（提交是有一定不可逆性的操作，按会话规则确认后再做）。

**卡住超 60 分钟就降级**（换笨实现或写进 Known Limitations），不要死磕。

---

## 5. 当前项目状态（2026-08-19 更新）

- **尚未 `git init`**——STEP0 的环境搭建部分还没做，这是接下来第一件事
- `.venv` 已创建，解释器 **Python 3.12.14**，与 playbook 一致（playbook 已从 3.11 改为 3.12）
- `data/target.pdf` 已就位（约 8.4MB，PDF 1.5，zip deflate 编码），**尚未做雷1/雷2/雷3
  三项检查**（文本层抽查、reranker 下载、embedding API 连通性），这些是 STEP0 的核心任务
- 项目骨架（`src/` `eval/` `config.yaml` 等）**尚未创建**
- 目前处于 **STEP0 开始前**的状态

进度请随实现推进更新本节（哪个 STEP/Slice 完成、关键数字如 Recall@5 等），
方便后续对话快速对齐上下文，不必每次重新翻 playbook 全文。

---

## 6. STEP 速查表（细节以 playbook 原文为准，此处只做导航）

| STEP | 内容 | RAG 环节 | 可否砍 |
|---|---|---|---|
| 0 | 环境 + 双模式骨架 + 拆三雷（文本层/reranker/API） | — | 否 |
| 1 | PDF 解析（PyMuPDF blocks，保存原始 page_no） | ① 解析 | 否 |
| 2 | 父子切分 + 双路索引(向量+BM25) + search()骨架 + 端到端问答 | ②③④⑤⑦⑧ | 否 |
| 3 | 手写 12 条评测题 + eval/harness.py（Recall@5/@20, MRR, Abstention） | 评测体系 | **绝对不能砍** |
| 4 | BM25+RRF 融合 / Cross-Encoder 重排，填 search() 两个 TODO | ④检索融合⑤精排 | 否 |
| 5 | small-to-big 父块扩展 + 页码引用生成 + 抗幻觉 prompt | ⑤扩展⑧生成 | 引用部分可简化 |
| 6 | dev→full 切换，补评测题到20条，出正式消融数据 | — | **不能砍** |
| 7 | FastAPI 服务 + README（含消融表、架构图、设计决策） | — | API可砍，README不能 |
| 8 | 收尾：简历数字、八问自测、可选参数扫描实验 | — | 可压缩 |

**四个消融配置**：`baseline`(全F) / `hybrid`(+bm25) / `rerank`(+bm25+rerank) /
`full`(+bm25+rerank+s2b)，对应 `search()` 的四个 if 分支逐步打开。

---

## 7. 关键机制备忘（跨 STEP 会反复用到）

- **`{suffix}` 机制**：`config.dev.enabled=true` → suffix=`"_dev"`，否则 suffix=`""`，
  所有 `config.paths.*` 和 `collection_name` 里的占位符据此替换，是 dev/full 隔离的唯一保障
- **`load_config(overrides: dict)`**：点号路径覆盖任意配置项，同时服务于"消融实验切开关"
  和"dev/full 模式切换"两个场景
- **模块级单例 + `reset_clients()`**：chroma client / bm25 / parents 映射都用单例避免重复加载，
  但必须能在 STEP6 切换模式后重置，否则会读到旧集合的数据
- **tokenize 一致性**：`src/ingest/tokenizer.py` 的分词函数必须被 indexer（建索引）和
  search（BM25查询）**同一处调用**，绝不能各写一套，否则 BM25 完全失效
- **RRF 用排名不用原始分数**融合（向量分数和 BM25 分数量纲不同，直接加权不稳定）
- **STEP3 后 `eval/harness.py` 和 `eval/golden_qa.jsonl` 锁死**，之后任何 STEP 的提示词
  都要带"严禁修改 eval/"

---

## 8. 参考

- 详细的每一步"①原理 ②手动任务 ③完整提示词"：见
  [Askingdoc_dev_playbook_v2.md](Askingdoc_dev_playbook_v2.md)（同目录）
- 每次实现新 STEP 前，先定位 playbook 里对应章节，把提示词内容当作实现规格，
  不要凭记忆改写，尤其【改动范围】和【严禁修改】部分要逐字遵守

## 9. 其他规则

- 每次启动前必读claude.md
- 每一轮对话，请在提示词以外，读取以下内容：
【项目背景】
Askingdoc：长文档 RAG 问答系统，个人项目，一周工期。
技术栈：Python 3.11 / PyMuPDF / LlamaIndex(仅切分) / ChromaDB / rank_bm25 /
       embedding API / bge-reranker-v2-m3 / FastAPI
设计原则：
- 检索链路手写，不用框架高层抽象
- 四个消融开关由 config.yaml 控制，一份代码跑四种配置
- 双数据集模式：dev(前150页) 与 full(全量) 共用同一份代码，
  仅由 config.dev.enabled 切换，禁止写两套逻辑

【代码规范】
- 注释和 docstring 用英文，要具体不要复述函数名
- 路径用 pathlib
- 关键中间结果打印出来便于调试
- 生成后请逐行解释关键逻辑的工作原理
