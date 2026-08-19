# Askingdoc 开发执行手册 v2

> 小数据集开发 · 全量交付
> 9 个 STEP，7 天闭环。配套 `Askingdoc_1week_spec.md`（规格书）。

---

## 核心流程变更说明

v1 的做法是选一本 300 页 PDF 从头用到尾。v2 改成**双数据集模式**：

```
DEV 模式（STEP 0–5）          FULL 模式（STEP 6–8）
前 150 页                     全量 500 页
索引重建 ~1 分钟              索引重建 ~5 分钟
12 条评测题                   20 条评测题
独立 chroma 集合 _dev         正式集合
用途：快速试错                用途：出正式数据、交付
```

**为什么这样做**：STEP 2 你大概率要重建索引 2–3 次（切分参数不对、metadata 漏字段、page_no 映射有 bug）。每次省下 4 分钟，一周累积能省一小时，而且心理负担完全不同——重建索引只要一分钟时，你才敢放心改切分策略。

**关键约束**：两种模式**共用同一份代码**，只由 `config.yaml` 的 `dev.enabled` 开关切换。绝不允许出现"dev 版代码"和"正式版代码"两套——那样 STEP 6 切全量时会出一堆意外。

---

## 使用说明

每个 STEP 三块内容：

- **① 这步在干什么** —— 目标、原理、在 RAG 全局的位置
- **② 你要手动做什么** —— AI 干不了或不该干的事
- **③ 给落地 AI 的提示词** —— 直接复制，已按"一次跑一个完整模块"的粒度设计

### 全局纪律

1. **每个 STEP 前后各 `git commit` 一次**。
2. **提示词里的"改动范围"不要删**——防止 AI 一次改十个文件导致跑不通又回不去。
3. **STEP 3 之后 `eval/harness.py` 锁死**。评测代码一变，消融数字失去可比性。
4. **AI 交付代码后先自己读一遍再运行**，看不懂就追问。目标是面试能讲，不是代码能跑。
5. **单个问题卡超 60 分钟就降级**：换笨实现，或写进 Known Limitations。

### 全局提示词头部（每次都带）

```
【项目背景】
Askingdoc：长文档 RAG 问答系统，个人项目，一周工期。
技术栈：Python 3.12 / PyMuPDF / LlamaIndex(仅切分) / ChromaDB / rank_bm25 /
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
```

---

# STEP 0｜环境搭建 + 双模式骨架

**耗时**：1.5–2h ｜ **D1 上午**

## ① 这步在干什么

搭环境、建骨架、**拆三颗雷**，并把双数据集机制在配置层就设计好。

三颗雷是一周计划里唯一能导致推倒重来的风险，必须在写业务代码前排除：

- **雷 1：PDF 没文本层**。扫描件里的"文字"是图片，提取出来是空的。整个项目建立在能提取文本的前提上。
- **雷 2：reranker 下载失败**。bge-reranker-v2-m3 约 2.3GB，国内网络可能很慢。等到 STEP 4 才发现就来不及。
- **雷 3：embedding API 不通**。账号/余额/网络任一环节有问题都会卡住 STEP 2。

**双模式的设计要点**：dev 和 full 必须用**不同的 chroma 集合名和不同的 bm25 文件路径**。否则切换时会污染彼此的数据，出现"明明只索引了 150 页却检索出第 400 页内容"这种诡异 bug。

## ② 你要手动做什么

**（1）选定并准备 PDF**

要求：**有文本层**、**你熟悉的领域**、300–600 页。相机说明书是很好的选择（术语密集、你熟悉、场景贴近设备手册检索）。

放到 `data/target.pdf`，**不截取，用完整版**。截取由代码的 `--max-pages` 参数控制。

**（2）环境搭建**

```bash
mkdir askingdoc && cd askingdoc
git init
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

| 命令 | 作用 | 参数说明 |
|---|---|---|
| `python -m venv .venv` | 创建虚拟环境，隔离项目依赖 | `.venv` 是约定目录名，IDE 会自动识别 |
| `source .venv/bin/activate` | 激活环境，之后 pip 只装进这里 | Windows 用 `.venv\Scripts\activate` |

替代方案：`conda create -n askingdoc python=3.12`（已有 conda 习惯可用）；`uv venv`（更快，装了 uv 推荐）。

**（3）拆三雷**

```bash
pip install pymupdf sentence-transformers openai python-dotenv
```

雷 1 —— 文本层 + 页数：

```bash
python -c "
import fitz
doc = fitz.open('data/target.pdf')
print(f'总页数: {len(doc)}')
for p in [10, 80, 150, 300]:
    if p < len(doc):
        t = doc[p].get_text()
        print(f'--- 第{p}页, {len(t)} 字符 ---')
        print(repr(t[:150]))
"
```

用 `repr` 是为了看出乱码和不可见字符。**四页都有实际文字 → 通过。任何一页空白 → 立刻换 PDF。**

相机手册特有的检查：**额外抽一页菜单说明页**。菜单导航文本（`MENU → ⚙️ → [格式化]`）块很短且含图标符号，可能提取成乱码。少量乱码不影响大局，parser 的长度阈值调低即可，并在 README 记一条已知限制。

雷 2 —— 后台下 reranker（开另一个终端继续干活）：

```bash
export HF_ENDPOINT=https://hf-mirror.com     # Windows: set HF_ENDPOINT=...
python -c "
from sentence_transformers import CrossEncoder
m = CrossEncoder('BAAI/bge-reranker-v2-m3')
print(m.predict([('测试问题','测试文档')]))
"
```

30 分钟下不完就降级 `BAAI/bge-reranker-base`（1.1GB，效果略降，够用）。

雷 3 —— API 连通：

```bash
python -c "
from openai import OpenAI
c = OpenAI(api_key='你的key', base_url='你的base_url')
r = c.embeddings.create(model='BAAI/bge-m3', input=['测试'])
print(f'向量维度: {len(r.data[0].embedding)}')
"
```

**记下向量维度**（bge-m3 是 1024），STEP 2 要用。

## ③ 提示词

```
【全局提示词头部】

【本次任务】
从零创建 Askingdoc 项目骨架，重点是双数据集（dev/full）配置机制。

1. 目录结构（空目录放 .gitkeep）：
askingdoc/
├── src/  __init__.py
│   ├── ingest/     __init__.py
│   ├── retrieval/  __init__.py
│   ├── generation/ __init__.py
├── eval/ __init__.py
│   └── results/
├── data/
└── storage/

2. requirements.txt：
pymupdf, llama-index-core, chromadb, rank-bm25, jieba,
sentence-transformers, openai, python-dotenv, pyyaml, tqdm,
fastapi, uvicorn, rich

3. config.yaml —— 所有参数集中于此：

dev:
  enabled: true          # ★ 全局开关，STEP6 改成 false
  max_pages: 150         # dev 模式只处理前 N 页
  suffix: "_dev"         # dev 模式下所有产物路径/集合名加此后缀

paths:
  pdf: data/target.pdf
  blocks: storage/blocks{suffix}.jsonl
  parents: storage/parents{suffix}.jsonl
  children: storage/children{suffix}.jsonl
  chroma_dir: storage/chroma
  bm25: storage/bm25{suffix}.pkl

index:
  collection_name: askingdoc{suffix}

chunking:
  parent_size: 1500
  child_size: 300
  overlap: 50

retrieval:
  use_bm25: false
  use_rerank: false
  use_small_to_big: false
  rrf_k: 60
  recall_top_k: 20
  final_top_k: 5
  rerank_batch: 8

models:
  embedding: <模型名>
  embedding_dim: <维度>
  llm: <模型名>
  reranker: BAAI/bge-reranker-v2-m3

4. src/config.py —— 配置加载模块，这是本步的核心：
   - 读 config.yaml + .env（key 只从 .env 读，不写进 yaml）
   - 【关键】{suffix} 占位符解析：
     dev.enabled=true  时 suffix = dev.suffix（"_dev"）
     dev.enabled=false 时 suffix = ""（空字符串）
     所有 paths 和 collection_name 里的 {suffix} 自动替换
   - 【关键】提供 load_config(overrides: dict = None)
     支持运行时覆盖任意配置项（点号路径，如 "retrieval.use_bm25"）
     这是后面消融实验和 dev/full 切换的基础
   - 用 dataclass 或 pydantic 提供类型提示
   - 提供 config.is_dev 属性和 config.describe() 方法
     （打印当前模式、页数限制、集合名，每次运行都要能一眼看清在哪个模式）

5. .gitignore：.venv/ data/ storage/ .env __pycache__/ *.pkl

6. README.md 骨架（只写章节占位）：
   简介 / 架构 / 快速开始 / 消融实验结果 / 设计决策 / 已知限制 / Future Work

【改动范围】
首次创建，可创建上述所有文件。不要写任何业务逻辑
（parser/chunker/search 留到后续 STEP）。

【输出后请解释】
1. {suffix} 机制是怎么实现的，为什么 dev 和 full 必须用不同的
   chroma 集合名和 bm25 路径（不隔离会发生什么）
2. load_config 的 overrides 机制怎么工作，
   它同时服务于哪两个场景（消融实验 / 模式切换）
3. 为什么把所有参数集中到 config.yaml 而非散在各模块
```

**验收**：

```bash
pip install -r requirements.txt
python -c "from src.config import load_config; c=load_config(); c.describe()"
```

应打印出 dev 模式、150 页限制、集合名 `askingdoc_dev`。

再验证一次切换：`load_config({"dev.enabled": False}).describe()` 应显示 full 模式和集合名 `askingdoc`。**这个验证必须做**，它是后面 STEP 6 一键切换的保障。

**commit**：`step0: skeleton with dev/full dual-dataset config`

---

# STEP 1｜PDF 解析层

**耗时**：1.5h ｜ **D1 下午**

## ① 这步在干什么

PDF → 一行一条的 JSON，每条带**页码**。这是 RAG 流程第 ① 步。

全项目最关键的约束在这里：

> **`page_no` 必须在这一步保存。一旦丢失，后续无法从文本反推页码，引用溯源这个核心卖点整个做不了。**

这是唯一一个"后补代价极大"的字段，面试也能讲。

用 PyMuPDF 的 `get_text("blocks")` 模式——返回版面块（一段、一个标题各一块），比整页纯文本保留更多结构，也方便按位置过滤页眉页脚。

**dev 模式在这里生效**：只解析前 `max_pages` 页。注意页码要用**原始页码**，不能因为截取就从 1 重编——否则 STEP 6 切全量时评测集的 `gold_pages` 全部错位。

## ② 你要手动做什么

**人工抽查 3 个块**：挑三条记录，翻 PDF 到对应页，确认文字确实在那页。

**这个验证必须做。** 页码错位是"跑起来没报错、但整个项目建立在错误基础上"的 bug，越晚发现越贵。

看统计：正常量级是**每页 3–15 块**。150 页解析出 100 个块说明过滤太狠；解析出 5000 个说明切太碎。

## ③ 提示词

```
【全局提示词头部】

【当前进度】
STEP 0 完成：骨架、config.yaml、src/config.py 就位。
当前 dev.enabled=true, max_pages=150。
config.paths 内容：<粘贴>

【本次任务】
实现 src/ingest/parser.py，PDF → 带页码的文本块。

功能要求：
1. 用 PyMuPDF 打开 config.paths.pdf
2. 【dev 模式】只处理前 config.dev.max_pages 页
   （dev.enabled=false 时处理全部）
   支持 --max-pages 命令行参数覆盖
3. 【页码规则】page_no 用 PDF 原始页码（PyMuPDF 从 0 开始，输出时 +1）。
   即使 dev 模式只处理前 150 页，页码也必须是真实页码，
   绝不能因为截取而重新编号——否则切全量时评测集会全部错位。
   请在代码注释里写明这条约束。
4. 逐页 page.get_text("blocks")，输出：
   {"block_id":"b_00001", "page_no":87, "text":"...", "bbox":[x0,y0,x1,y1]}
5. 噪声过滤：
   - 丢弃长度 < 20 字符的块
   - 丢弃纯数字/纯符号块
   - 丢弃位于页面顶部5%或底部5%区域且长度<50的块（用 bbox 判断）
   - 阈值都从 config 读，方便调整
6. 文本清洗：
   - 合并块内断行（PDF 换行多是排版换行非语义换行）
   - 归一化连续空白
   - 保留段落语义边界
7. 输出 JSONL 到 config.paths.blocks
8. 打印统计报告：处理页数 / 保留块数 / 过滤块数 / 平均块长 /
   最长最短块长 / 每页平均块数 / 当前模式（dev or full）
9. debug 函数 preview_page(page_no)：打印指定页所有块，
   用于人工核验页码

【改动范围】
只新建 src/ingest/parser.py。不要修改 config.py。
需要新增配置项就在回复里说明，我手动加。

【输出后请解释】
1. get_text("blocks") 返回的元组每个位置是什么，
   为什么用 blocks 而非 "text" 或 "dict" 模式
2. bbox 坐标系原点在哪，怎么用它判断页眉页脚
3. 断行合并逻辑：怎么区分"排版换行"和"段落结束"
```

**验收**：跑通、每页 3–15 块、抽查 3 块页码正确。

**commit**：`step1: PDF parser with original page numbering`

---

# STEP 2｜切分 + 双路索引 + 打通问答

**耗时**：3–4h（dev 模式比 v1 快约 1h）｜ **D2**

## ① 这步在干什么

做完这步项目就能问答了。含 RAG 流程的 ②③④⑤⑦⑧ 六步。

**父子两级切分（small-to-big）** 是本项目唯一稍精巧的设计。矛盾在于：检索要求块**小**（噪声少、语义集中、命中精确），生成要求块**大**（上下文完整）。解法是切两级——parent 约 1500 tokens，child 约 300 tokens，child 记住 `parent_id`。**只有 child 进向量库**，命中后换回 parent 喂 LLM。

**双路索引**：child 同时进 ChromaDB 和 BM25。今天两个都建好，但检索只用向量——BM25 检索逻辑留到 STEP 4，这样 STEP 4 不用回头改 ingest 层。

**为什么今天就把 `search()` 的四个 if 分支骨架搭好**：消融实验要求"一份代码 + 配置开关"。先写死向量检索、后面再改造，容易引入无关差异。骨架先立，后面只填分支。

**dev 模式的收益在这里最明显**：150 页约 250 个 child，embedding 一分钟内跑完。你可以放心地重建 2–3 次。

## ② 你要手动做什么

**（1）分三次 slice 提交**，中间出问题能回滚。

**（2）embedding 阶段盯进度**。dev 模式约 250 条、4 个 batch，应该很快。**如果发现调用次数远超预期（比如上百次），立刻 Ctrl+C 检查是不是循环写错了。**

**（3）跑通后自己问 5 个问题**。不是测指标（那是 STEP 3），是**建立手感**：检索出的段落和问题相关吗？答案有没有明显胡编？这个手感后面调优时很有用。

## ③ 提示词

### Slice 2-1：父子切分

```
【全局提示词头部】

【当前进度】
STEP 1 完成。storage/blocks_dev.jsonl 已生成，前 2 行真实数据：
<粘贴>
统计：处理 150 页，XXX 个块，平均块长 XXX 字符

【本次任务】
实现 src/ingest/chunker.py，small-to-big 父子两级切分。

功能要求：
1. 读 config.paths.blocks（dev/full 由 config 自动决定）
2. Parent 层：顺序累积 blocks 到接近 config.chunking.parent_size
   - 实现 estimate_tokens(text)：中文约 1 字符=1 token，
     英文约 4 字符=1 token，请处理混合文本
   - parent 记录：{"pid":"p_0012","text":"...","pages":[86,87],
                   "block_ids":[...]}
   - pages 是覆盖的所有页码，去重升序
3. Child 层：对每个 parent 文本用 LlamaIndex SentenceSplitter
   切成 child_size，overlap 用 config 值
   - child 记录：{"cid":"c_0045","parent_id":"p_0012",
                  "text":"...","page_no":87}
4. 【最关键约束】child 的 page_no 必须正确。
   parent 可能跨页，实现方式：拼 parent 文本时记录每个 block
   在 parent 内的字符区间，child 切出后按其起始偏移落在哪个区间
   来确定 page_no。请在代码里详细注释这段逻辑。
5. 输出 config.paths.parents / config.paths.children
6. 统计：parent 数 / child 数 / 平均每 parent 的 child 数 /
   child 平均长度 / 跨页 parent 比例 / 当前模式
7. verify_page_mapping(n=10)：随机抽 n 个 child，
   打印 text 前 50 字和 page_no，供人工核验

【改动范围】
只新建 src/ingest/chunker.py。

【输出后请解释】
1. SentenceSplitter 的 chunk_overlap 做什么，为什么需要重叠，
   50 是什么考虑
2. child 的 page_no 继承逻辑，用一个跨页的具体例子走一遍
3. small-to-big 的数据结构怎么支撑"子块检索、父块生成"
```

跑 `verify_page_mapping()` 抽查，**commit**。

### Slice 2-2：向量化 + 双路索引

```
【全局提示词头部】

【当前进度】
chunker 完成，children_dev.jsonl 有 XXX 条。
embedding 模型 <名>，向量维度 <数字>。
当前 collection_name = askingdoc_dev

【本次任务】
实现 src/ingest/indexer.py，建立向量索引和 BM25 索引。

功能要求：
1. 读 config.paths.children
2. 向量索引：
   - openai 库调 embedding API（base_url 从 config，key 从 .env）
   - 批量调用 batch_size=64
   - 存入 ChromaDB PersistentClient，集合名从 config.index.collection_name
     （dev 模式自动是 askingdoc_dev，与全量集合物理隔离）
   - metadata 必须含：cid, parent_id, page_no
   - 【断点续传】启动时查已有 cid，跳过已入库的
   - 【重试】失败时指数退避重试 3 次
   - tqdm 进度条
3. BM25 索引：
   - 实现统一的 tokenize(text)：中文 jieba.lcut，英文按空格，
     混合文本统一处理
   - 【重要】这个 tokenize 函数请放在 src/ingest/tokenizer.py 单独模块，
     因为 STEP 4 的检索侧必须复用完全相同的分词逻辑
   - BM25Okapi 建索引
   - pickle {"cids":[...], "corpus_tokens":[...]} 到 config.paths.bm25
     （只存分词结果，运行时重建 BM25Okapi 对象，减小文件体积）
4. CLI 参数：
   --rebuild   清空重建（默认增量）
   --limit N   只处理前 N 条（调试用）
5. 统计：入库数 / 跳过数 / 失败数 / 预估 token 消耗 / 耗时 / 当前模式和集合名

【改动范围】
新建 src/ingest/indexer.py 和 src/ingest/tokenizer.py。

【重要】
请先支持 --limit 20 让我小规模验证，确认无误我再跑全量。

【输出后请解释】
1. 为什么 embedding 要批量调用，batch_size 的权衡是什么
2. ChromaDB metadata 有什么用，为什么必须存 parent_id 和 page_no
3. BM25 原理（词频、逆文档频率、长度归一化），
   它和向量检索各自在什么场景更强
```

**先 `--limit 20` 验证，再跑 dev 全量。commit。**

### Slice 2-3：检索骨架 + CLI

```
【全局提示词头部】

【当前进度】
askingdoc_dev 集合已有 XXX 条，bm25_dev.pkl 已生成。

【本次任务】
实现检索层骨架和 CLI，打通端到端。
本次只实现向量检索分支，其余三个开关留 TODO。

1. src/retrieval/search.py —— 项目核心
   search(query: str, cfg) -> List[Hit]
   Hit 是 dataclass：{cid, text, page_no, score, parent_id}

   函数结构严格按此骨架（这是消融实验的基础）：

   def search(query, cfg):
       hits = vector_search(query, top_k=cfg.recall_top_k)
       if cfg.use_bm25:
           pass    # TODO STEP4: BM25 召回 + RRF 融合
       if cfg.use_rerank:
           pass    # TODO STEP4: Cross-Encoder 精排
       hits = hits[:cfg.final_top_k]
       if cfg.use_small_to_big:
           pass    # TODO STEP5: child -> parent 扩展
       return hits

   本次实现 vector_search。
   资源（chroma client、embedding client）用模块级单例。
   【注意】单例要能感知 config 变化——STEP6 切全量时集合名会变，
   请提供 reset_clients() 函数或用 config 指纹做缓存 key。

2. src/generation/prompts.py：
   ANSWER_PROMPT，硬性要求：只基于给定片段回答、
   不得使用自身知识、找不到就回复"文档中未提及"。
   引用格式先留占位，STEP5 强化。

3. src/generation/answerer.py：
   answer(query, cfg) -> {"answer": str, "hits": List[Hit]}

4. src/cli.py：
   交互式循环问答，显示答案 + 检索到的页码。
   支持 :config 打印当前配置（含 dev/full 模式）、:quit 退出。

【改动范围】
新建 search.py / prompts.py / answerer.py / cli.py。
不要修改 ingest/ 下的文件。

【关键设计约束】
search() 必须返回结构化 Hit 列表，绝不能返回拼好的字符串。
原因：STEP3 的评测脚本需要"检索出的 top-k 及其页码"这个中间结果。

【输出后请解释】
1. search() 的四分支骨架为什么这样设计，
   它和"一份代码+配置开关做消融"的关系
2. 模块级单例怎么实现，为什么检索场景需要它，
   以及 reset_clients 为什么在双模式下是必需的
3. prompt 里"不得使用自身知识"在 RAG 里为什么重要
```

**验收**：`python -m src.cli` 能问答，自己问 5 个问题建立手感。

**commit + tag**：

```bash
git commit -m "step2: chunking, dual index, baseline retrieval"
git tag baseline-dev
```

---

# STEP 3｜评测集 + Eval Harness

**耗时**：3h（dev 版 12 条，比 v1 少）｜ **D3**

## ① 这步在干什么

**今天不写功能代码，只建评测体系。这是全项目最不该跳过的一天。**

没有评测集，STEP 4–5 的所有优化都是凭感觉。面试官问"你怎么知道重排真的有用"，你只能说"感觉好了一些"——项目就白做了。有评测集，你能说"在自建评测集上，加入重排后 Recall@5 从 62% 提升到 78%"。两个量级的回答。

**v2 的评测集分两批写**：

- **今天写 12 条**，全部落在**前 150 页范围内**（dev 模式检索不到 150 页之后的内容）
- **STEP 6 补到 20 条**，新增的 8 条覆盖全书后半部分

**指标含义**（面试要用）：

- **Recall@5**：top-5 里包含正确页码的问题占比。主指标。
- **Recall@20**：召回上限。和 Recall@5 的差值很有意思——差值大说明"召回够了但排序差"（该加重排），差值小说明"召回本身不足"（该调切分或加 BM25）。**这个分析在面试里很出彩。**
- **MRR**：首个正确结果排名的倒数均值。第 1 位得 1.0，第 3 位得 0.33。衡量排序质量。
- **Abstention Rate**：无答案问题上正确拒答的比例。抗幻觉指标。

## ② 你要手动做什么

**手写 12 条 golden QA（今天）。全项目唯一不能交给 AI 的一步。**

**为什么不能让 AI 生成**：AI 从 chunk 反向生成的问题会与 chunk 高度同源——用词、句式、信息组织都像原文。这种问题检索必然命中，Recall 虚高，反映不了真实能力。你自己提的问题会用你自己的说法，这才是真实用户的样子。

格式：

```json
{"qid":"q001","question":"连拍模式下缓冲区能存多少张RAW？","gold_pages":[87],"gold_answer":"约23张","type":"fact_lookup","in_dev_range":true}
```

`in_dev_range` 字段很重要——STEP 6 补题后，harness 要能按模式筛选。

**dev 批 12 条配比**：

| 类型 | 条数 | 说明 |
|---|---|---|
| `fact_lookup` | 6 | 单点事实，答案在一个块内 |
| `multi_hop` | 2 | 需合并两处信息，`gold_pages` 填多个 |
| `keyword_exact` | 2 | 含型号/菜单项/参数代号，**专测 BM25** |
| `negative` | 2 | 文档中不存在答案，`gold_pages` 为空 |

相机手册的 `keyword_exact` 很好写：具体镜头型号、菜单项全名、按钮代号。这类正是向量检索的弱项、BM25 的强项——**你的消融表第 2 行大概率会有明显提升，有对比效果的实验比没效果的好讲。**

`negative` 怎么编：问同品牌其他机型的参数，或问手册确实没写的内容（"这台相机支持8K吗"而手册只写到4K）。

**两个提问技巧**：不要用文档原句当问题，用你自己的说法重述；故意混几个口语化问法（"连拍能拍多少张不卡"），真实用户就这么问。

## ③ 提示词

```
【全局提示词头部】

【当前进度】
STEP 2 完成，dev 模式端到端跑通，tag baseline-dev 已打。
search(query, cfg) -> List[Hit]，Hit 字段：cid, text, page_no, score, parent_id

我已手写 eval/golden_qa.jsonl，当前 12 条（全部在前150页范围内），格式：
{"qid","question","gold_pages","gold_answer","type","in_dev_range"}
type: fact_lookup(6) / multi_hop(2) / keyword_exact(2) / negative(2)
negative 类 gold_pages 为空数组。
STEP6 会补到 20 条，新增的 in_dev_range=false。

【本次任务】
实现 eval/harness.py —— 检索质量评测框架。
这是项目核心资产，写完我会锁死此文件，请一次做完整。

功能要求：

1. 加载 golden_qa.jsonl
   【关键】按当前模式过滤：
   config.dev.enabled=true  时只跑 in_dev_range=true 的题
   config.dev.enabled=false 时跑全部
   打印实际参与评测的题数

2. 对每条非 negative 问题：
   - 调 search(question, cfg)
   - 【注意】要拿到 top-20 用于算 Recall@20，
     不能被 final_top_k 截断——请在 harness 内临时覆盖 final_top_k
   - 命中判定：hit.page_no 在 gold_pages 里

3. 指标：
   - Recall@5  = 前5个结果至少命中一个 gold_page 的问题数 / 总数
   - Recall@20 = 同上取前20
   - MRR       = mean(1/首个命中排名)，未命中记 0
   - 分 type 统计（便于分析 BM25 对 keyword_exact 的贡献）

4. negative 类单独处理：
   - 调完整 answer() 拿生成结果
   - 判定 abstention：答案含"未提及/没有提到/未找到/无法回答"等模式
   - 输出 abstention_rate

5. 四组预设配置（写在 eval/configs.py）：
   baseline: bm25=F, rerank=F, s2b=F
   hybrid:   bm25=T, rerank=F, s2b=F
   rerank:   bm25=T, rerank=T, s2b=F
   full:     bm25=T, rerank=T, s2b=T
   通过 config.py 的 overrides 机制注入

6. 输出：
   - JSON 存 eval/results/{mode}_{config_name}.json
     mode 是 dev 或 full —— 【重要】两种模式的结果文件必须分开，
     否则 STEP6 全量重跑会覆盖 dev 数据，无法对照
     内容含：总体指标 + 分type指标 + 每题明细 + 元信息(模式/题数/时间)
   - 终端打印格式化表格（用 rich）

7. eval/compare.py：
   读 results/ 下指定模式的所有 json，打印四行对比表，
   并输出 markdown 表格（可直接粘进 README）
   支持 --mode dev|full

8. --verbose 模式：打印每题检索到的 top-5 页码 vs gold_pages，
   用于人工分析失败原因

【改动范围】
只新建 eval/ 下文件（harness.py, configs.py, compare.py）。
【严禁修改 src/ 下任何文件】

【输出后请解释】
1. MRR 的计算方式，它和 Recall@5 分别衡量什么，为什么两个都要看
2. Recall@20 和 Recall@5 的差值能说明什么问题
3. 为什么 negative 类要单独走生成流程而非只看检索
4. 为什么 dev 和 full 的结果文件必须分开存
```

**跑第一组**：

```bash
python -m eval.harness --config baseline
```

记下 dev 版 baseline 的三个指标。**注意：dev 数据只有参考价值，正式数字要等 STEP 6。**

## ⚠️ 本步后 `eval/harness.py` 锁死

之后所有提示词写"禁止修改 eval/"。评测代码一变，四组数字失去可比性。

**commit**：`step3: eval harness and dev golden QA set`

---

# STEP 4｜混合检索 + 重排

**耗时**：3–4h ｜ **D4**

## ① 这步在干什么

填 `search()` 的两个 TODO，产出消融表第 2、3 行（dev 版）。代码改动都不大，一天能做两组。

**BM25 + RRF 解决什么**：向量检索理解语义，但对**精确字符串**弱。问"FE 200-600mm 镜头的最近对焦距离"，向量检索会把所有镜头的对焦距离段落都拉回来——"型号X的对焦距离"这个句式语义都差不多。BM25 靠词项精确匹配，对专有名词准得多。

**RRF 公式**：

```
score(d) = Σ_r  1 / (k + rank_r(d))     k 取 60
```

关键点：**用排名不用原始分数**。向量相似度（0–1）和 BM25 分数（无上界）量纲完全不同，直接加权需要归一化和调权重，很脆弱。用排名绕开了这个问题——**这是 RRF 最大优点，面试可以讲。**

**Cross-Encoder 重排解决什么**：向量检索是 Bi-Encoder（双塔）——问题和文档**分别**编码再比距离，快，但两者从未真正交互。Cross-Encoder 把问题和文档**拼在一起**送进模型，能捕捉细粒度对应关系，精度高得多。

代价是慢：每个候选都要过一遍模型，全库跑不现实。所以**两阶段架构**：Bi-Encoder 粗筛 20，Cross-Encoder 精排 5。这是搜索引擎和推荐系统的通用套路——**说得出这个类比就显得你理解了架构本质。**

## ② 你要手动做什么

**（1）每加一个模块跑一次 eval，立刻记录。** 不要两个都做完再跑，中间数据丢了补不回来。

**（2）用 `explain_fusion` 看失败案例。** 挑几道从"未命中"变"命中"的题——通常就是 keyword_exact 那两条。**亲眼看到 BM25 救回了哪道题**，这个观察比数字更帮你理解，面试也能当例子讲。

**（3）某模块没提升不要慌**，见下面的分析框架。

## ③ 提示词

### Slice 4-1：BM25 + RRF

```
【全局提示词头部】

【当前进度】
STEP3 完成，eval harness 已锁定。
dev baseline：Recall@5=XX%, Recall@20=XX%, MRR=XX

src/retrieval/search.py 当前内容：
<粘贴完整文件>

storage/bm25_dev.pkl 结构：{"cids":[...], "corpus_tokens":[...]}
分词函数在 src/ingest/tokenizer.py

【本次任务】
实现 search() 的 use_bm25 分支。

1. bm25_search(query, top_k) -> List[Hit]
   - 加载 config.paths.bm25，运行时重建 BM25Okapi（模块级单例）
   - query 用 src/ingest/tokenizer.py 的【同一个】tokenize 函数分词
     （绝对不能自己另写一套，不一致会导致 BM25 完全失效）
   - get_scores 取 top_k
   - cid 反查 text/page_no/parent_id：
     加载 children.jsonl 建内存映射

2. rrf_fuse(dense_hits, sparse_hits, k) -> List[Hit]
   - score(d) = Σ_r 1/(k + rank_r(d))，rank 从 1 开始
   - 两路都出现的文档自然得高分
   - 按新分数降序，保留 Hit 原始字段
   - k 从 config.retrieval.rrf_k 读

3. 接进 search() 的 use_bm25 分支

4. debug 函数 explain_fusion(query)：
   打印两路各自 top-10（cid+页码+排名）及融合后 top-10，
   标注每个结果来自哪一路、原排名多少。
   我要用它直观理解 RRF 效果。

【改动范围】
只修改 src/retrieval/search.py，可新增 src/retrieval/fusion.py。
【严禁修改 eval/ 下任何文件】
【严禁修改 config.yaml 既有键】

【输出后请解释】
1. RRF 为什么用排名而非原始分数融合，用分数加权会遇到什么问题
2. k=60 的作用，调大调小分别有什么影响
3. tokenize 一致性为什么是死要求，不一致会发生什么
```

跑 `python -m eval.harness --config hybrid`，**记第 2 行**。用 `explain_fusion` 看两道 keyword_exact。commit。

### Slice 4-2：Cross-Encoder 重排

```
【全局提示词头部】

【当前进度】
dev hybrid：Recall@5=XX%, MRR=XX（相比 baseline 变化 XX）
search.py 当前内容：<粘贴>
reranker 已在 STEP0 预下载：BAAI/bge-reranker-v2-m3

【本次任务】
实现 search() 的 use_rerank 分支。

1. sentence-transformers 的 CrossEncoder 加载 reranker
   - 【模块级单例】，绝不能每次查询都加载
   - 首次加载打印耗时提示

2. rerank(query, hits) -> List[Hit]
   - 构造 [(query, hit.text) for hit in hits] 送 model.predict
   - batch_size 从 config.retrieval.rerank_batch 读，默认 8
   - 按新分数降序，新增 rerank_score 字段，保留原 score 便于对比

3. 接进 search() 的 use_rerank 分支

4. debug 函数 explain_rerank(query)：
   打印重排前后 top-10 对比（页码+原排名+新排名+两个分数），
   标出排名变化最大的几条

5. 打印单次 rerank 耗时。CPU 上超过 3 秒的话，
   在回复里给降级建议（减少候选数 / 换 bge-reranker-base）

【改动范围】
只修改 src/retrieval/search.py，可新增 src/retrieval/reranker.py。
【严禁修改 eval/】

【输出后请解释】
1. Bi-Encoder 和 Cross-Encoder 的结构区别，
   为什么 Cross-Encoder 精度更高但不能用于全库检索
2. "召回-排序"两阶段架构在搜索/推荐系统里的普遍性，
   为什么几乎所有检索系统都这么设计
3. 重排候选数(20)怎么权衡，太大太小分别什么问题
```

跑 `--config rerank`，**记第 3 行**。

**分析框架**（结论写进 README）：

| 观察 | 说明 | 怎么写 |
|---|---|---|
| 重排提升明显 | 召回够了，原排序不准 | "验证了两阶段架构的必要性" |
| 重排提升很小 | Recall@20≈Recall@5，召回本身是瓶颈 | "瓶颈在召回阶段，后续应优化切分策略" |
| BM25 提升明显 | 文档专有名词密集 | "关键词通道对精确匹配类查询贡献显著" |
| BM25 几乎无提升 | keyword_exact 太少，或文档以语义型内容为主 | "本场景下语义检索已覆盖多数查询" |

**任何结果都能写成有深度的结论。** 能解释"为什么这个优化在我的场景下无效"，比一路顺风更能证明你理解系统。

**commit**：`step4: hybrid retrieval with RRF and cross-encoder rerank`

---

# STEP 5｜父块扩展 + 引用生成

**耗时**：3–4h ｜ **D5**

## ① 这步在干什么

上午填最后一个 TODO，dev 版四行数据齐全。

下午做引用溯源——这是项目对国企/制造业场景的核心卖点。「这个回答的依据在手册第几页」是实际业务必问的问题，一个说不出依据的 AI 回答，在需要合规和审计的场景里没有价值。

**small-to-big 的实际动作**：精排出的 top-5 child，按 `parent_id` 换成 parent。注意**去重**——多个 child 命中同一 parent 很常见（说明该 parent 高度相关），只留一份，分数取最高。

**抗幻觉两手段**：prompt 硬约束 + 引用强制。配合 STEP 3 那 2 条 negative 样本，就是完整方案。

## ② 你要手动做什么

跑完 full 配置，**dev 版四行齐全**——先别填简历，等 STEP 6 的全量数字。

**人工抽检引用准确率**（10 条）：挑 10 个问题，核对答案里标的 `[p.87]` 翻到 PDF 第 87 页内容是否对得上。这一步只能人工做，20 分钟，值得。

**测 2 条 negative**：看系统是否老实说"未提及"。开始编造就回去加强 prompt。

## ③ 提示词

### Slice 5-1：small-to-big

```
【全局提示词头部】

【当前进度】
dev rerank：Recall@5=XX%, MRR=XX
search.py 已实现三个分支。
parents_dev.jsonl 结构：{"pid","text","pages":[...],"block_ids":[...]}

【本次任务】
实现 search() 的 use_small_to_big 分支，最后一个 TODO。

1. 加载 config.paths.parents 建 pid -> parent 内存映射（模块级单例，
   同样需要支持 reset 以适应 STEP6 的模式切换）

2. expand_to_parents(hits) -> List[Hit]
   - 精排后的 top-k child 按 parent_id 换成 parent 内容
   - 【去重】多个 child 命中同一 parent 时只留一条，分数取最高
   - 【补足】去重后数量会少于 final_top_k，
     从剩余候选按分数补足，保持最终数量稳定
   - Hit 的 text 换成 parent 文本，新增 pages 字段，
     保留 page_no 作为主页码

3. 接进 search() 的 use_small_to_big 分支

4. debug 函数 explain_expansion(query)：
   打印扩展前后对比——原 child 数、去重后 parent 数、补足几条、
   上下文总长度变化

5. 【兼容性】eval/harness.py 已锁定不能改。
   请确保扩展后 Hit 仍有 page_no 字段且语义不变（harness 用它判定命中）。
   有冲突就在回复里告诉我，不要擅自改 eval。

【改动范围】
只修改 src/retrieval/search.py，可新增 src/retrieval/expander.py。
【严禁修改 eval/】

【输出后请解释】
1. small-to-big 具体解决什么矛盾，不做会怎样（举一个跨块答案的例子）
2. 去重逻辑为什么必要，不去重浪费什么
3. 扩展后上下文变长，对 LLM 生成有什么正反两面影响
```

跑 `--config full`，记第 4 行。跑 `python -m eval.compare --mode dev` 出完整对比表。commit。

### Slice 5-2：引用生成 + 抗幻觉

```
【全局提示词头部】

【当前进度】
dev 四组消融完成：baseline/hybrid/rerank/full 的 R@5 分别为 XX/XX/XX/XX
检索层已完整，本次只改生成层。

【本次任务】
实现页码级引用溯源和抗幻觉约束。

1. 重写 src/generation/prompts.py：
   - 上下文格式：每片段前标 [来源: 第X页] 或 [来源: 第X-Y页]
   - 硬性要求 A：只基于提供片段回答，不得使用模型自身知识
   - 硬性要求 B：每个事实性陈述后标注来源，格式 [p.87]，
     多来源写 [p.87][p.92]
   - 硬性要求 C：找不到答案时只回复"文档中未提及相关信息"，
     禁止推测、禁止用常识补充
   - 硬性要求 D：不确定时宁可说不知道
   - 写成清晰的 system prompt，并给 few-shot 示例
     （一个正常回答 + 一个正确拒答）

2. src/generation/answerer.py：
   - 拼上下文时带页码标注
   - 返回增加 cited_pages（从答案正则解析出的页码列表）
   - 增加 has_answer 布尔字段

3. src/cli.py：
   - 用 rich 高亮 [p.N]
   - 答案下方列"参考来源"：页码 + 该页片段前 80 字
   - 拒答时给明确提示

4. 新增 src/generation/verify.py：
   spot_check(n=10)：随机抽 n 条 golden QA 跑完整流程，
   输出人工核验清单 markdown 表格：
   | 问题 | 生成答案 | 答案标注页码 | gold_pages | 需人工确认 |

【改动范围】
修改 prompts.py / answerer.py / cli.py，新增 verify.py。
【严禁修改 search.py 和 eval/】

【输出后请解释】
1. 为什么"不得使用自身知识"在 RAG 里是必须的，不加会怎样
2. few-shot 为什么能提升指令遵循，拒答示例的作用
3. 引用溯源在企业文档问答场景的实际价值
```

**commit**：`step5: small-to-big and cited generation`

---

# STEP 6｜全量切换与正式数据 ★ 新增

**耗时**：2–2.5h ｜ **D6 上午**

## ① 这步在干什么

**这是 v2 独有的一步，也是整个项目"从开发到交付"的转折点。**

前五步你在 150 页的小数据集上把所有逻辑跑通了。现在把 `dev.enabled` 改成 `false`，让同一份代码跑全量 500 页，重新产出**正式的四行消融数据**。

**为什么这一步是安全的**：因为 STEP 0 就把双模式做进了配置层，所有产物路径和集合名由 `{suffix}` 自动隔离。理论上改一个布尔值 + 重跑 ingest 就完事。

**为什么还是要留 2 小时**：理论和实际之间总有意外——单例没重置、评测集页码超出范围、某个路径硬编码了。这一步的价值就是**把这些意外在交付前暴露出来**。

**这一步在面试里也有话讲**："我用小数据集开发、全量交付，这样迭代快且最终指标真实"——这是实际工程里很常见的做法（用采样数据开发 pipeline，全量跑生产），说出来显得你有工程习惯。

## ② 你要手动做什么

**（1）补足 golden QA 到 20 条**

新增 8 条，全部落在**第 150 页之后**，`in_dev_range` 设 `false`。配比：

| 类型 | 新增 | 累计 |
|---|---|---|
| `fact_lookup` | 4 | 10 |
| `multi_hop` | 2 | 4 |
| `keyword_exact` | 1 | 3 |
| `negative` | 1 | 3 |

相机手册后半部分通常是菜单详解、自定义设置、规格表。**避开规格表页出题**（多列表格解析不好，这是你 README 里已经写明的已知限制），集中在菜单说明和操作步骤部分。

**（2）执行切换**

```bash
# 1. 改 config.yaml
#    dev:
#      enabled: false

# 2. 全量 ingest（约 5-8 分钟）
python -m src.ingest.parser
python -m src.ingest.chunker
python -m src.ingest.indexer

# 3. 四组消融重跑
python -m eval.harness --config baseline
python -m eval.harness --config hybrid
python -m eval.harness --config rerank
python -m eval.harness --config full

# 4. 出正式对比表
python -m eval.compare --mode full
```

**（3）对照 dev 与 full 的数据**

这是个有价值的观察：全量数据下指标通常会**略降**（干扰项变多，检索难度上升）。降多少？如果 Recall@5 从 dev 的 85% 掉到 full 的 70%，说明规模效应明显，这个现象值得写进 README——**它证明你的评测不是在玩具数据上刷出来的**。

**（4）重新人工抽检引用准确率**（10 条，全量版）

**（5）填简历数字**——用 full 的数字，不是 dev 的。

## ③ 提示词

只在切换出问题时才需要 AI 介入。先自己跑，遇到报错再用这个：

```
【全局提示词头部】

【当前进度】
STEP0-5 完成，dev 模式下四组消融数据齐全。
现在执行 dev → full 切换：config.yaml 的 dev.enabled 改为 false。

遇到的问题：
<粘贴完整报错 / 异常现象>

当前配置：<粘贴 config.yaml>
相关代码：<粘贴报错涉及的文件>

【本次任务】
排查并修复 dev→full 切换的问题。

【排查方向提示】
1. 模块级单例是否缓存了 dev 的 chroma client / bm25 / parents 映射，
   切换后未重置
2. 是否有硬编码路径绕过了 config 的 {suffix} 机制
3. eval/harness 的 in_dev_range 过滤逻辑是否正确处理了 full 模式
4. children.jsonl 的内存映射是否加载了正确的文件

【改动范围】
只修复问题本身，不要顺便重构。
【严禁修改 eval/harness.py 和 eval/golden_qa.jsonl】
（这两个文件任何改动都会让已有数据失去可比性）

【修复后请说明】
问题的根本原因是什么，为什么在 dev 模式下没暴露出来
```

**验收**：

- `eval/results/` 下有 `full_baseline.json` 等四个文件
- `compare --mode full` 输出四行完整数据
- CLI 能问到第 150 页之后的内容（**这是切换成功最直观的验证**——随便问一个后半部分的问题，dev 模式下必然答"未提及"）

**commit**：`step6: switch to full dataset, official ablation results`

---

# STEP 7｜API 服务 + README

**耗时**：3h ｜ **D6 下午**

## ① 这步在干什么

**上午**：包一层 HTTP 服务。不是为了部署，是为了**面试时能展示**——FastAPI 自带的 `/docs` 交互页截图比贴代码直观得多。

**下午**：写 README。这是**面试官唯一会看的东西**。

一个残酷事实：面试官不会 clone 你的仓库跑起来，也不会逐个文件读代码。他会花 60 秒扫一遍 README。所以前 30 行必须出现：项目做什么、架构图、**四行消融表**。

## ② 你要手动做什么

**（1）启动服务，用 `/docs` 问 2–3 个问题，截图。** 挑效果好的（引用清晰、答案准确）——这是你的门面。

**（2）README 的消融表亲自填。** 不要让 AI 从 json 读，它可能读错或编造。你对着 `eval/results/full_*.json` 填。

**（3）Known Limitations 认真写。** 主动声明限制是加分项。被问"扫描件怎么办"，答"已知限制，需接 PaddleOCR 做文本层回退，评估约 2 天，不在本次范围"——比支支吾吾好得多，显示你做过技术评估而非没想到。

相机手册特有的两条限制值得写进去：**图标类 UI 符号无法解析**、**多列规格表结构丢失**。这两条是你实打实遇到的，比通用限制有说服力。

## ③ 提示词

```
【全局提示词头部】

【当前进度】
STEP6 完成，全量正式数据：
| 配置 | Recall@5 | Recall@20 | MRR |
| baseline | XX | XX | XX |
| hybrid   | XX | XX | XX |
| rerank   | XX | XX | XX |
| full     | XX | XX | XX |
引用准确率（人工抽检10条）：XX%
Abstention rate（3条negative）：XX%
文档规模：XXX 页，XXX 个 parent，XXX 个 child

【本次任务】
实现 HTTP 服务并生成 README。这是交付步骤。

第一部分：src/api.py

1. FastAPI 端点：
   POST /ask
     请求：{"question": str, "config": str = "full"}
     响应：{"answer": str,
            "sources":[{"page":int,"snippet":str,"score":float}],
            "has_answer": bool, "latency_ms": int}
   GET /health  -> {"status":"ok","index_size":int,"mode":"full"}
   GET /config  -> 当前生效配置

2. 【启动预加载】用 FastAPI lifespan，
   启动时加载 chroma/bm25/reranker/parents 映射，
   不要在请求里加载（否则首个请求要等几十秒）

3. Pydantic 请求响应模型，每个字段写 description
   （这样 /docs 页面好看，截图才有说服力）

4. 全局异常处理，返回结构化错误

5. 文件头注释写清启动命令

第二部分：README.md 完整版

结构：
1. 标题 + 一句话简介
2. 核心特性（3-4条，突出：页码级引用溯源 / 混合检索 / 消融验证）
3. 架构图（ASCII 画出离线 ingest 和在线 query 两条流水线）
4. 【消融实验结果】四行表格，数字我填，你留占位
5. 快速开始：环境要求 / 安装 / .env 配置 / 三条命令
6. 项目结构（目录树 + 每模块一行说明）
7. 【开发方法】说明双数据集模式：
   小数据集(150页)快速迭代，全量(XXX页)出正式指标，
   共用一份代码由配置开关切换
8. 设计决策（重点，写详细，每条说清"问题→方案→代价"）：
   - 为什么 RAG 而非长上下文（成本对比 + lost in the middle）
   - 为什么混合检索（向量对精确字符串弱 + RRF 用排名的好处）
   - 为什么两阶段召回-排序（Bi vs Cross Encoder）
   - 为什么 small-to-big（检索精度 vs 生成完整性的矛盾）
   - 评测方法论（自建评测集 + 单变量消融 + 一份代码配置开关）
9. 已知限制（表格：限制项/原因/后续方案）：
   扫描件OCR、多列表格结构、UI图标符号、多文档路由、单元测试
10. Future Work

【改动范围】
新建 src/api.py，重写 README.md。
【严禁修改 src/retrieval/ 和 eval/】

【README 写作要求】
- 前30行内必须出现：项目定位、架构图、消融表
- 设计决策每条说清"问题是什么→怎么解决→代价是什么"
- 不要空话（"采用先进技术"这类删掉）
- 中文为主，技术术语保留英文
```

**验收**：`uvicorn src.api:app --reload` 启动，`/docs` 能问答，截图存 README。

**commit**：`step7: FastAPI service and README`

---

# STEP 8｜收尾与交付

**耗时**：2–3h ｜ **D7**

## ① 这步在干什么

缓冲日。**必然会用上**——前七步总有某处超时（依赖冲突、PDF 怪字符、API 限流、切换全量出意外）。不要提前挪用。

一切顺利的话，做三件提升面试表现的事。

## ② 你要手动做什么

**（1）简历数字填实**（用 full 数据）

```latex
\item \textbf{项目成果：}自建 20 条评测集（含 3 条无答案样本用于抗幻觉验证），
完成四组消融实验量化各检索模块贡献，Recall@5 由 62\% 提升至 81\%，
引用页码准确率 90\%
```

LaTeX 里 `%` 写成 `\%`。

**（2）对着八问自测（最重要）**

关上文档，口头回答：

1. 什么是 RAG，为什么不直接把文档丢给大模型？
2. 为什么要混合检索？RRF 为什么用排名不用分数？
3. 重排是什么，为什么不用它检索全部？
4. small-to-big 解决什么问题？
5. **你怎么验证这些优化真的有用？**（把四行数字报出来）
6. chunk_size 为什么选这个值？
7. 这个系统有什么局限？
8. **你的开发流程是怎样的？**（讲双数据集：小数据快速迭代、全量出正式指标、一份代码配置切换）

第 5 题是核心，**四行数字背下来**。第 6 题若没做扫描实验，诚实答"当前是经验值 512，这是我打算下一步用评测集验证的点"——诚实且显示你知道该怎么验证。第 8 题是 v2 新增的加分项。

**（3）录 30 秒 CLI 演示**（录屏或 GIF），放 README 顶部。

## ③ 提示词（可选，有余力才做）

```
【全局提示词头部】

【当前进度】
项目主体完成，README 已写，全量四组消融数据齐全。
今天是缓冲日，想补参数扫描实验增加深度。

【本次任务】
实现 eval/sweep.py —— 参数扫描实验。

1. rrf_k 扫描：20/60/100（不用重建索引，很快）
2. 重排候选数扫描：top-10/20/50（也不用重建）
3. chunk_size 扫描：256/512/1024（需重建索引，最有价值也最贵）
   - 用独立集合名 askingdoc_sweep_{size} 避免污染正式索引
   - 每组跑一次 full 配置 eval

4. 输出 markdown 表格，可直接粘进 README 的"补充实验"一节

【改动范围】
只新建 eval/sweep.py。
【严禁修改 eval/harness.py 和 eval/golden_qa.jsonl】

【注意】
chunk_size 扫描需重新 embedding，产生约 3 倍于一次全量的 API 费用。
代码里加确认提示，不要直接跑。

【输出后请解释】
chunk_size 影响检索质量的机制，为什么太大太小都不好
```

**最终**：`git commit -m "step8: final polish"` + `git tag v1.0`

---

# 交付检查表

- [ ] 可复现 git 仓库，README 前 30 行有定位、架构图、消融表
- [ ] **全量数据的**四行消融表（不是 dev 数据）
- [ ] `eval/golden_qa.jsonl` 20 条手写（含 3 条 negative）
- [ ] 引用准确率数字（全量版人工抽检 10 条）
- [ ] FastAPI 可启动，`/docs` 截图在 README
- [ ] Known Limitations 每条写清原因和后续方案
- [ ] README 有"开发方法"一节说明双数据集流程
- [ ] 简历数字填实，标注"至今"
- [ ] **能口头答完 STEP 8 的八个问题**

最后一项没做到，前八项价值打对折。项目本身不会说话，你才会。

---

# 附录：时间账与优先级

| STEP | 内容 | 时长 | 天 | 可否砍 |
|---|---|---|---|---|
| 0 | 环境 + 双模式骨架 + 拆雷 | 1.5–2h | D1 上 | 否 |
| 1 | PDF 解析 | 1.5h | D1 下 | 否 |
| 2 | 切分 + 索引 + 跑通 | 3–4h | D2 | 否 |
| 3 | **评测集 + harness** | 3h | D3 | **绝对不能砍** |
| 4 | 混合检索 + 重排 | 3–4h | D4 | 否 |
| 5 | small-to-big + 引用 | 3–4h | D5 | 引用部分可简化 |
| 6 | **全量切换 + 正式数据** | 2–2.5h | D6 上 | **不能砍** |
| 7 | API + README | 3h | D6 下 | API 可砍，README 不能 |
| 8 | 收尾 | 2–3h | D7 | 可压缩 |

合计约 23 小时。

**严重超时时砍的顺序**：STEP 8 扫描实验 → STEP 7 的 API → STEP 5 的 verify.py。

**STEP 3 和 STEP 6 无论如何不能砍。** STEP 3 没有评测集，STEP 4–5 的工作全部失去意义；STEP 6 不切全量，简历上的数字就是 150 页玩具数据跑出来的——这个心虚在面试被追问"你的文档多大"时会立刻暴露。

---

# 附录：dev/full 差异速查

| 项 | dev | full |
|---|---|---|
| `dev.enabled` | true | false |
| 页数 | 前 150 | 全部 |
| blocks 文件 | `blocks_dev.jsonl` | `blocks.jsonl` |
| chroma 集合 | `askingdoc_dev` | `askingdoc` |
| bm25 文件 | `bm25_dev.pkl` | `bm25.pkl` |
| 评测题 | 12 条（in_dev_range=true） | 20 条（全部） |
| 结果文件 | `dev_*.json` | `full_*.json` |
| ingest 耗时 | ~1 分钟 | ~5–8 分钟 |
| 用途 | 快速迭代 | 正式指标、交付 |

**任何时候不确定当前在哪个模式，跑 `python -c "from src.config import load_config; load_config().describe()"`。**
