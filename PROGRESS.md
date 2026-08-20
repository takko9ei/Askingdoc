# Askingdoc — 进度文档

> 动态文档，每完成一个 STEP/Slice 就更新。规则性内容见 [CLAUDE.md](CLAUDE.md)，
> 技术决策的完整论述见 [docs/DECISIONS.md](docs/DECISIONS.md)。
> 本文件最后更新：2026-08-20（STEP4 完成后）。

---

## 1. 进度总览表

| STEP | 内容 | 状态 | 完成日期 | commit |
|---|---|---|---|---|
| 0 | 环境 + 双模式骨架 | ✅ 已完成 | 2026-08-20 | `6267742` first commit |
| 1 | PDF 解析 | ✅ 已完成 | 2026-08-20 | `7bf0281` PDF parser with original page numbering |
| 2 | 切分 + 双路索引 + 跑通CLI（Slice 2-1/2-2/2-3 全部完成） | ✅ 已完成 | 2026-08-20 | `aba7bcc` "123"（含chunker/indexer/tokenizer/search/prompts/answerer/cli） |
| 3 | 评测集(12条) + `eval/harness.py` | ✅ 已完成，**已锁定** | 2026-08-20 | `9dbf30f` evaluate step setup |
| 4 | BM25+RRF融合（Slice 4-1）+ Cross-Encoder重排（Slice 4-2） | ✅ 已完成 | 2026-08-20 | `615b6fc` slice 4: rerank finished |
| 5 | small-to-big父块扩展 + 页码引用生成 | ⬜ 未开始 | — | — |
| 6 | dev→full切换 + 评测题补到20条 + 正式消融数据 | ⬜ 未开始 | — | — |
| 7 | FastAPI服务 + README完整版 | ⬜ 未开始（README目前只有STEP0建的章节占位） | — | — |
| 8 | 简历数字 + 八问自测 + 可选参数扫描 | ⬜ 未开始 | — | — |

**注**：playbook 建议每个 STEP 后打 git tag（如 `baseline-dev`），实际开发中用户选择只用
commit message 区分阶段，未打任何 tag（`git tag -l` 为空）。commit 粒度也不是严格
一个 STEP 一个 commit（STEP2 三个 slice 合并成一次提交）。这是实际发生的情况，不是本文档的错误。

---

## 2. 当前状态快照

- **当前处于**：STEP4 完成之后，STEP5 开始之前
- **下一步具体任务**：STEP5 Slice 5-1（small-to-big 父块扩展），实现 `search()` 的
  `use_small_to_big` 分支——这是 `search()` 四个消融分支里最后一个还没填的 TODO
- **当前模式**：dev（`config.dev.enabled=true`，前150页，独立集合 `askingdoc_dev`）。
  尚未切换到 full 模式，`eval/results/full_*.json` 不存在

---

## 3. 已完成模块清单

> 下一个 AI 可以直接 import 调用这些函数，不需要重读全部源码。每条格式：
> 路径 / 职责 / 关键函数签名 / 已验证通过的点。

### `src/config.py` — 配置加载
职责：读 `config.yaml` + `.env`，解析 `{suffix}` 占位符，提供运行时 override 机制。
```python
load_config(overrides: dict[str, Any] | None = None) -> Config
Config.is_dev -> bool
Config.describe() -> None   # 打印当前模式快照
```
已验证：`load_config()` 默认返回 dev 模式（`askingdoc_dev`）；
`load_config({"dev.enabled": False})` 正确切到 full 模式（`askingdoc`）；
override 一个不存在的 key（如打错字）会抛 `KeyError`，不会静默失败。

### `src/ingest/parser.py` — PDF解析
职责：PyMuPDF `get_text("blocks")` 逐页解析，噪声过滤，输出带原始页码的 JSONL。
```python
parse_pdf(cfg: Config) -> list[dict]
preview_page(page_no: int, cfg: Config | None = None) -> None   # 调试用，打印某页原始block
```
已验证：150页解出873块，覆盖137个不同页码（部分页全部块被过滤掉，正常现象）；
页码抽查通过，且第83页原文页脚本身印着"83"，与算出的 `page_no` 吻合（独立佐证）。
噪声过滤阈值（`MIN_BLOCK_CHARS=20`等）是模块级常量，**未按最初设想迁移进
`config.yaml`**，见第5节技术债。

### `src/ingest/chunker.py` — 父子两级切分
职责：blocks → parent（顺序累积到~1500token）→ child（LlamaIndex SentenceSplitter切~300token）。
```python
estimate_tokens(text: str) -> float   # 中文≈1字符1token，英文≈4字符1token
build_index_units(cfg: Config) -> tuple[list[dict], list[dict]]   # 主入口
verify_page_mapping(n: int = 10, cfg: Config | None = None) -> None   # 调试：抽查child页码
```
已验证：29个parent，205个child，跨页parent占96.6%（这本手册block极碎，正常现象）；
用具体跨页案例（p_0012横跨60-66页）人工走查过child页码继承逻辑，单调递增，正确。

### `src/ingest/tokenizer.py` — BM25统一分词
```python
tokenize(text: str) -> list[str]   # jieba.lcut + 小写 + 丢弃纯符号token
```
**全项目唯一分词实现，见 CLAUDE.md【铁律】。**

### `src/ingest/indexer.py` — 建索引
职责：child → ChromaDB向量索引 + BM25分词语料pickle。
```python
build_vector_index(cfg: Config, children: list[dict], rebuild: bool = False) -> dict
build_bm25_index(cfg: Config, children: list[dict]) -> dict
```
已验证：205条全部入库（185新增+20跳过+0失败），断点续传、批量embedding、重试逻辑均实测过。

### `src/retrieval/search.py` — 检索核心
职责：项目最核心的模块，`search()` 是唯一检索入口，四个消融分支的骨架。
```python
Hit  # dataclass: cid, text, page_no, score, parent_id, rerank_score(默认None)
vector_search(query: str, cfg: Config, top_k: int) -> list[Hit]
bm25_search(query: str, cfg: Config, top_k: int) -> list[Hit]
search(query: str, cfg: Config) -> list[Hit]   # 主入口，四分支：vector(必) / bm25(可选) / rerank(可选) / small_to_big(TODO)
reset_clients() -> None
explain_fusion(query: str, cfg=None) -> None   # 调试：看BM25+向量融合细节
explain_rerank(query: str, cfg=None) -> None   # 调试：看重排前后排名变化
```
已验证：`use_small_to_big` 仍是 `pass` TODO，其余三个分支都已实测。
模块级单例（chroma client/collection、embedding client、bm25 index、children map）
均按"配置指纹"缓存，dev/full切换安全。

### `src/retrieval/fusion.py` — RRF融合
```python
rrf_fuse(dense_hits: list[Hit], sparse_hits: list[Hit], k: int) -> list[Hit]
```
`k` 从 `config.retrieval.rrf_k` 读（当前值60）。用排名不用原始分数融合，见DECISIONS.md第3条。

### `src/retrieval/reranker.py` — Cross-Encoder重排
```python
rerank(query: str, hits: list[Hit], cfg: Config) -> list[Hit]
reset_reranker() -> None
```
已验证：本地加载耗时约6.5-6.8秒（模型已在STEP0下载好，这只是读入内存的时间）；
20个候选重排耗时约2.3秒；曾发现并修复一个真实bug——BM25融合后候选数未裁回
`recall_top_k`，导致同时开BM25+重排时实际重排38条而非20条（已在`search()`和
`explain_rerank()`两处修复）。

### `src/generation/prompts.py` / `src/generation/answerer.py` — 生成
```python
ANSWER_PROMPT: str   # 硬约束：只用给定片段、不用自身知识、找不到就说"文档中未提及相关信息"
format_context(hits: list[Hit]) -> str
answer(query: str, cfg: Config) -> dict   # {"answer": str, "hits": list[Hit]}
```
引用格式目前是自然语言（"第87页提到..."），STEP5会强化成 `[p.87]` 格式的强制标注。

### `src/cli.py` — 交互式CLI
```python
main() -> None   # python -m src.cli
```
支持 `:config`（打印当前配置）、`:quit`（退出）。已实测交互循环、真实问答。

### `eval/golden_qa.jsonl` — 评测集（🔒已锁定）
12条，配比 fact_lookup(6)/multi_hop(2)/keyword_exact(2)/negative(2)，全部
`in_dev_range=true`。8条用户手写，4条AI补充（已知情确认存在"同源"风险，见CLAUDE.md第7节）。

### `eval/harness.py` / `eval/configs.py` / `eval/compare.py`（🔒harness已锁定）
```python
run_evaluation(config_name: str, verbose: bool = False) -> dict   # python -m eval.harness --config X
ABLATION_CONFIGS: dict[str, dict]   # baseline/hybrid/rerank/full 四组override预设
```
已验证：四组配置全部真实跑通，结果见第4节。

---

## 4. 关键数据现状

| 项 | 值 |
|---|---|
| 目标PDF | `data/target.pdf`，576页，中文为主（含英文型号名，如"ILCE-7CM2 α7CII"） |
| blocks_dev.jsonl | 873条（前150页范围，覆盖137个不同页码，平均块长53.6字符，min20/max613） |
| parents_dev.jsonl | 29条（跨页比例96.6%） |
| children_dev.jsonl | 205条（平均长度260.3字符） |
| chroma集合 | `askingdoc_dev`，205条，向量维度**1024**（bge-m3） |
| bm25_dev.pkl | 205条文档 |
| eval/golden_qa.jsonl | 12条（dev批次，STEP6会补到20条） |

### 四组消融数据（dev模式，最新一次真实运行结果）

| 配置 | Recall@5 | Recall@20 | MRR | Abstention |
|---|---|---|---|---|
| baseline | 90.0% | 100.0% | 0.710 | 100.0% |
| hybrid | **100.0%** | 100.0% | 0.637 | 100.0% |
| rerank | 90.0% | 100.0% | 0.641 | 100.0% |
| full | 90.0% | 100.0% | 0.641 | 100.0% |

原始数据在 `eval/results/dev_{baseline,hybrid,rerank,full}.json`。`rerank`和`full`
数字目前完全一致——符合预期，两者唯一差异项`use_small_to_big`仍是TODO，STEP5填完才会分化。
**这份数据只是dev模式的开发期参考，不是最终交付数字**——STEP6切到full模式后要重新跑一遍
正式数据，简历/README用的是full数字，不是这份。

详细的逐题排名分析、假阳性案例、bug修复记录见 [docs/capability_log.md](docs/capability_log.md)
(4个时间节点的记录，从STEP2跑通CLI起持续追踪)。

---

## 5. 待办与已知问题

### 下一步任务
STEP5 Slice 5-1：实现 `search()` 的 `use_small_to_big` 分支——精排后的 top-k child
按 `parent_id` 换成完整 parent 内容，去重（多个child命中同一parent时只留一条、分数取最高），
去重后数量不足要从剩余候选补足。之后 Slice 5-2 做引用格式强化（`[p.87]`格式）+ 抗幻觉prompt。

### 遇到过但未解决的问题
1. **"如何使用触摸功能图标进行拍摄"重排后排名从5跌到13**——候选池里一堆内容都在讲
   "触摸功能图标"（隐藏图标/播放画面图标/拍摄画面图标……），Cross-Encoder把这些同主题
   内容都判成高相关（分数普遍0.78~0.97），没能精细区分出"具体拍摄步骤"这个更窄的意图。
   **待验证**：STEP5的small-to-big扩展到完整parent上下文后，Cross-Encoder是否更容易判别。
2. **"如何更换相机电池"的生成综合能力疑点**——检索到了含答案的片段(page29)，但生成的
   回答只用了另一个片段(page44)，没有整合page29的内容。不确定是LLM综合能力不够，
   还是page29本身就不是标准的"更换电池"步骤（细读原文更像是"充电故障排查"语境）。
   留到STEP5/生成质量优化时判断。
3. **第112-113页的多列超链接跳转目录被PyMuPDF解析成一整块乱麻文本**——已确认是PDF
   解析层的结构性限制（多个独立菜单项名称被硬拼在一起，无分隔符），不是索引或检索的bug。
   考虑写进STEP7 README的"已知限制"表格。

### 技术债 / 临时妥协
1. `src/retrieval/search.py` 里"运行召回阶段"的逻辑（`vector_search` + 可选`bm25_search`+
   `rrf_fuse`）在 `search()`、`explain_fusion()`、`explain_rerank()` 三处重复实现。
   曾因此产生过一次真实bug（`recall_top_k`截断只在`search()`修了，`explain_rerank()`
   漏了，导致调试工具和实际检索路径行为不一致），已发现并修复，但重复代码本身还在，
   值得未来抽成一个共享私有函数。
2. `src/ingest/parser.py` 的噪声过滤阈值（`MIN_BLOCK_CHARS=20`、`HEADER_FOOTER_MARGIN_PCT=0.05`、
   `HEADER_FOOTER_MIN_CHARS=50`）是模块级常量，最初设计是要迁移进`config.yaml`的`ingest:`段
   （需要同时给`src/config.py`加对应dataclass），用户表示会手动加，但截至目前没有跟进，
   常量还留在`parser.py`里。
3. 目前没有单元测试（pytest等），全靠手动运行+人工核验。playbook本身把这条列为可接受的
   已知限制（个人一周项目的合理取舍），计划写进STEP7 README的已知限制表。

---

## 6. 环境备忘

**Python**：3.12.14（playbook原定3.11，STEP0搭建时环境已是3.12，未回退，两者兼容无冲突）
**venv路径**：项目根目录下 `.venv/`（`.gitignore`已排除）

**启动命令**：
```bash
# 激活环境
source .venv/bin/activate

# ingest 三步（全量跑，会自动按 config.dev.enabled 决定处理范围）
python -m src.ingest.parser          # PDF -> storage/blocks_dev.jsonl
python -m src.ingest.chunker         # blocks -> parents/children
python -m src.ingest.indexer         # children -> chroma + bm25.pkl
python -m src.ingest.indexer --limit 20   # 小规模验证用

# 交互式问答
python -m src.cli

# 评测（四组配置）
python -m eval.harness --config baseline   # 也可 hybrid/rerank/full
python -m eval.harness --config rerank --verbose   # 看每题细节
python -m eval.compare --mode dev          # 四组横向对比表
```

**`.env` 需要的变量**（只列名字，不写值，真实值只在本地`.env`里，已gitignore）：
```
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
LLM_API_KEY=
LLM_BASE_URL=
```
`.env.example` 里有同样的占位模板和注释说明。当前实际使用的服务商：embedding走硅基流动
（`https://api.siliconflow.cn/v1`），LLM走DeepSeek官方API（`https://api.deepseek.com`）——
这两个值写在这里是因为它们不是密钥本身（base_url不是敏感信息），仅供环境复现参考。

**踩过的环境坑及解法**：
1. `pymupdf` 新版本里 `import fitz` 已标记废弃（会打印warning），改用
   `import pymupdf as fitz` 消除警告，用法不变。
2. venv一开始只装了STEP0"拆三雷"用到的几个包（pymupdf/sentence-transformers/openai/
   python-dotenv），到STEP2才发现`llama-index-core`/`chromadb`/`rank-bm25`/`jieba`/
   `fastapi`/`uvicorn`都没装——补跑一次 `pip install -r requirements.txt` 即可，
   之后没再遇到过依赖缺失问题。
3. 曾有一次调用DeepSeek API时TLS握手阶段被reset（`Connection reset by peer`），
   诊断为执行环境的瞬时网络问题（不是DNS也不是超时，`curl -v`显示ClientHello发出去
   立刻被断开），重试后即恢复正常，不是代码bug，不需要特殊处理，但如果再次遇到
   同类问题，先用`curl -v https://api.deepseek.com/`排查是不是网络层面被拦截。
4. `jieba`首次调用会向stderr打印"Building prefix dict from the default dictionary..."
   之类的准备日志（词典加载，约0.2~0.3秒），属于正常行为，不是错误。
5. reranker（`BAAI/bge-reranker-v2-m3`）在STEP0"拆三雷"阶段已下载到本地缓存，
   之后每次调用只是"加载到内存"（约6.5~6.8秒），不是重新下载；如果换了机器/清了缓存，
   会退化成真实下载（约2.3GB，playbook建议超过30分钟下不完就降级用
   `BAAI/bge-reranker-base`，1.1GB，效果略降）。
