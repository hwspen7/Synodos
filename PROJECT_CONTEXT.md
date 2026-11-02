# Meeting of Pioneers 项目完整上下文

> 用途：将本文件提供给 GPT 客户端，作为理解、讨论和继续开发本项目的基础资料。
> 整理日期：2026-11-02。依据当前工作目录内的业务源码和配置静态分析编写。
> 本次没有启动服务、执行模型推理或运行构建；下文的“已实现”表示代码中存在实现，不代表已通过完整运行验收。未读取或导出已有会议、声纹库的私有内容。

## 1. 项目是什么

这是一个以中文会议为主要场景的本地语音转写 Web 应用，产品名称为 **Meeting of Pioneers**。用户在浏览器打开麦克风，后端识别语音、恢复标点、提取声纹，并为转写片段标注说话人。会议可以保存、继续记录，导出 Markdown 或 JSON。

项目的三类核心资产是：

- **会议记录**：逐句文字、时间戳、说话人标签、声纹特征及会议配置。
- **说话人声纹库**：一个人可以拥有多条声纹，用于在后续会议中自动标注姓名。
- **热词库**：保存人名、产品名、行业术语等，在会议开始时选择并传入识别模型。

当前形态是前后端分离、本地模型推理、文件持久化的应用。业务代码没有调用 OpenAI API，没有接入大语言模型自动总结，也没有实现账号体系、数据库或云端同步。

## 2. 已有功能与使用流程

### 2.1 会议

首页展示会议列表，以及累计时长、片段数、说话人数等统计。可以新建、查看、继续、删除和导出会议。累计说话人数是各会议说话人数求和，并非跨会议去重的人数。

新建会议时输入标题，前端先创建记录，再进入会议页并尝试自动开启麦克风。查看历史会议时也会建立 WebSocket，以恢复已有片段；仅查看本身不会必然开始录音。

会议页提供：

- 开始、停止录音；显示连接状态、麦克风电平和缓冲时长。
- 片段视图与段落视图切换、自动滚动、复制全文。
- 调整处理阈值、确认阈值；选择参与者、热词库和声纹匹配阈值。
- 手动保存、停止后的自动保存、导出 MD / JSON。
- 停止后再次开始，可恢复同一会议并继续记录。

参与者和热词设置通过“应用并重连”生效，录音中不能应用。修改热词只影响后续识别，不会重新识别已经确认的文字；修改参与者或匹配阈值后，历史片段的说话人标签会重新计算。

**“清空”按钮只清空前端当前显示，不会删除后端片段或磁盘会议。** 后续完整快照或重连可能再次显示原内容。

### 2.2 说话人声纹库

可创建、改名、删除说话人，给同一人追加多条声纹，或删除单条声纹。声纹有两种来源：

1. 浏览器现场录音，上传 PCM 后由后端提取特征。
2. 从已保存会议中选择一个或多个片段，直接复用片段已有的声纹向量。

录音少于 0.2 秒会被拒绝；默认少于 5 秒的向量标记为 `short` 并提示可能不准确，但并不因此禁止保存或匹配。会议导入是每个片段各存一条向量，不是把多段音频合并后重新提取。

### 2.3 热词库

支持创建、编辑、删除热词库。热词会清理空白、去空串、保序去重。会议可选择多个库，合并去重后最多取 200 个词，以空格连接后传给 FunASR 的 `hotword` 参数。热词用于影响模型识别，不保证强制输出指定词。

## 3. 技术栈与目录

| 层次 | 当前技术 |
| --- | --- |
| 前端 | Vue 3、TypeScript、Vue Router、Vite、自定义 CSS |
| 录音 | getUserMedia、Web Audio、AudioWorklet、线性插值重采样 |
| 通信 | HTTP JSON、WebSocket 二进制 PCM 与 JSON 消息 |
| 后端 | Python ≥3.12、FastAPI、Uvicorn、asyncio、threading、NumPy |
| 识别 | FunASR 1.4.3、PyTorch 2.7.0、torchaudio 2.7.0 |
| 持久化 | Python pickle 文件，无数据库 |
| 依赖管理 | 后端 pyproject.toml + uv.lock；前端 package.json + pnpm-lock.yaml |

前端声明 Vue `^3.5.40`、Vue Router `^4.6.4`、Vite `^8.1.5`、TypeScript `~6.0.0`；这些是声明范围，精确解析以锁文件为准。FastAPI 声明 `>=0.141.1`，Uvicorn 声明 `>=0.52.4`。

前后端现在共用仓库根目录，以下是新的布局：

```text
meeting/
├─ main.py                         FastAPI 入口及路由注册
├─ api/
│  ├─ health.py                    运行状态
│  ├─ meetings.py                  会议保存、查询、导出
│  ├─ speakers.py                  声纹管理
│  ├─ hotwords.py                  热词管理
│  └─ websocket.py                 实时连接与协议
├─ core/
│  ├─ config.py                    环境配置
│  └─ runtime.py                   共享对象与会话注册表
├─ audio/buffer.py                 音频缓冲
├─ inference/models.py             FunASR 与聚类
├─ sessions/
│  ├─ session.py                   会话容器
│  └─ processor.py                 后台语音处理
├─ storage/
│  ├─ meetings.py                  会议文件持久化
│  ├─ speakers.py                  声纹持久化
│  └─ hotwords.py                  热词持久化
├─ example/                        手动验证脚本
├─ tests/                          后端基础测试
├─ src/                            Vue 前端源码
│  ├─ App.vue、main.ts
│  ├─ router/index.ts、types.ts
│  ├─ api/client.ts、api/ws.ts
│  ├─ audio/recorder.ts、pcm-processor.ts
│  ├─ speaker/clustering.ts
│  ├─ components/、pages/、assets/
├─ public/                         前端静态资源
├─ index.html、vite.config.ts      前端入口与开发代理
├─ package.json、pnpm-lock.yaml    前端依赖
├─ pyproject.toml、uv.lock         后端依赖
├─ tsconfig.json 等                TypeScript 配置
├─ README.md
└─ PROJECT_CONTEXT.md
```

依赖缓存、模型文件、日志、示例音频不是业务源码。仓库统一使用根目录 README.md，原先的空后端 README 和前端模板 README 已移除。

## 4. 系统架构和音频处理流程

```text
浏览器麦克风
  → AudioWorklet：单声道、重采样到 16kHz、量化 Int16
  → WebSocket /ws/audio：二进制 PCM
  → FastAPI Session → AudioBuffer
  → SessionProcessor 后台线程
      → ASR + VAD + 标点 + 句子时间戳
      → 筛选可信句子 → 提取 CAM++ 声纹
      → 已知参与者匹配 → 剩余片段全量聚类
      → transcript 完整快照
  → 浏览器：片段着色、段落合并、展示

手动保存 / 停止收尾 → MeetingStore → meeting.pkl
导出 → 从保存状态生成段落 → Markdown / JSON
```

### 4.1 录音格式

浏览器申请单声道音频，并启用回声消除、降噪和自动增益。AudioWorklet 将输入采样率按线性插值转换成 16000Hz，量化为 Int16 PCM，经主线程发送。

后端约定输入为 **16kHz、单声道、有符号 16 位小端、无 WAV 文件头的原始 PCM**，转换为 float32 时除以 32768。WebSocket 不负责自动解码 MP3、WAV 或其他压缩音频。

### 4.2 “实时”识别的实际含义

它是持续上传音频、按窗口批量识别的准实时转写，并非逐字流式 ASR。

1. 缓冲达到默认 10 秒，触发一轮识别。
2. 对当前整段缓冲调用 `transcribe_sentences()`，获取句子、标点和毫秒时间戳。
3. 默认只确认结束时间距缓冲末尾至少 2 秒的句子。
4. 对可信句子的音频提取一次声纹，保存为片段。
5. 裁掉已确认部分，未确认尾部保留到后续处理。
6. 文本与声纹确认后不会因标签重算而重新 ASR；尚未确认的尾部可能被多次识别。

这里的 `VAD_THRESHOLD_S` 是触发处理的缓冲时长，**不是模型判断是否有人说话的概率阈值**。`REC_THRESHOLD_S` 是句尾距缓冲尾的时间差，也不是 ASR 置信度分数。

默认配置下，不能把界面中“停顿 2 秒”的提示理解为两秒内必定出字：还取决于缓冲是否达到 10 秒、句子切分和推理耗时。

### 4.3 停止和续记

停止时前端先停麦，再发送 `{"type":"stop"}`。后端停止常规循环，处理剩余音频，此时不再要求句尾距缓冲尾满足确认阈值；默认不足 1 秒的剩余缓冲不做最终识别。随后保存状态、推送最终结果和 `closed` 消息。

续记恢复历史文字、声纹和标签配置，把新音频起点设置为最后一个已确认片段的结束采样点。时间轴是项目内部的累计片段时间轴，不会自动计入两次录音之间的现实时间间隔，上一轮未保存的尾部静音也不会完整保留。

## 5. 模型与说话人识别策略

### 5.1 模型

| 配置短名 | 角色 | 实际映射仓库 |
| --- | --- | --- |
| paraformer-zh | 中文 ASR | iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch |
| fsmn-vad | 语音活动检测、内部切分 | iic/speech_fsmn_vad_zh-cn-16k-common-pytorch |
| ct-punc-c | 标点恢复 | iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch |
| cam++ | 声纹 embedding | iic/speech_campplus_sv_zh-cn_16k-common |

`ModelManager` 统一封装模型。声纹向量会展开成一维 float32 并做 L2 归一化，代码说明通常为 192 维；业务存储不把维数固定写死。ASR 输出会清理汉字之间多余空白。

模型对象支持按需加载，但 FastAPI 启动事件会主动预加载 ASR、声纹和聚类，所以首次启动可能需要较长时间。代码没有显式设置推理设备，不能仅凭当前源码断言实际使用 CPU 或 GPU。

### 5.2 已知参与者与未知说话人

每个可信片段先与**本次选中的参与者**的全部声纹比较，取全局最大余弦相似度。达到阈值则标注该参与者姓名；不是对声纹取均值，也不是在整个声纹库中自动搜索所有人。

未匹配的片段交给 FunASR `ClusterBackend` 做全量重聚类。选择了参与者时显示“未知说话人N”，没有选择参与者时显示“说话人N”。后端根据新旧簇共享的片段集合映射稳定 ID，以减少显示跳变，但不保证标签永远不变，也不代表跨会议统一身份。

实际默认参与者匹配阈值为 **0.3**。`sessions/processor.py` 顶部说明中遗留的 0.5 与实现不一致，应以 `core/config.py` 和实际构造参数为准。参与者匹配阈值与聚类合并阈值 0.78 是两个不同参数。

### 5.3 段落合并

前端将按时间排列的相邻片段，在“同一 `speaker_id` 且间隔严格小于 `merge_gap_s`”时合并显示，默认 2 秒。文本直接连接，段落 ID 使用首个片段的 ID。

后端导出时用 `MeetingStore._build_paragraphs()` 生成相同规则的段落。合并只是展示与导出行为，不会合并音频重新 ASR。`core/config.py` 中关于合并后做 ASR 的旧注释不代表当前处理逻辑。

## 6. 并发与状态管理

- FastAPI 事件循环负责网络连接；耗时存储和部分等待通过 executor 执行。
- 每个 WebSocket 对应一个 Session 和一个守护线程 SessionProcessor。
- AudioBuffer 用锁与条件变量协调收音和处理。
- 所有会话共享一个 ModelManager；模型加载有锁，ASR、声纹和聚类推理共用全局推理锁，实际推理串行执行。
- 线程通过 `asyncio.run_coroutine_threadsafe()` 写入出站队列，再由异步转发任务发给浏览器；队列容量为 500。
- `sessions` 按 session_id 管理连接，`meeting_sessions` 按 meeting_id 索引活动会议。
- 同一会议出现新会话时会向旧处理器发停止信号；当前实现不属于多用户协同编辑协议。
- 内容指纹变化时才发送 transcript，每次发送的是**完整片段快照**；version 是会话内计数，不是全局数据库版本。

## 7. 数据保存与导出

默认路径相对于**后端进程启动时的工作目录**：

| 文件 | 内容 |
| --- | --- |
| meetings/<meeting_id>/meeting.pkl | `{"meta": ..., "state": ...}` |
| speakers/speakers.pkl | 说话人及多条声纹、来源、时长、时间、short 标记 |
| hotwords/hotwords.pkl | 热词库名称、词列表、创建与更新时间 |

会议 meta 包含 id、title、创建/更新时间、采样率、合并间隔、片段数、duration_ms 和说话人数。duration_ms 取最后片段结束位置，不能直接当作完整录音文件时长。

会议 state 保存参与者 ID、热词库 ID、匹配阈值、标签映射、next_segment_id，以及每个片段的：

```text
id、start_sample、end_sample、text、emb、cluster、speaker_id、speaker_name
```

内部时间以采样点为主，前端以毫秒为主：`毫秒 = 采样点 × 1000 / 16000`。会议 ID、说话人 ID、声纹 ID 通常是 UUID hex 字符串；片段 ID 和展示用 speaker_id 是整数，不能混用。

保存状态及声纹库、热词库更新主要通过临时文件加 `os.replace()` 替换。声纹库和热词库各自有进程内 RLock；这不等于支持多个后端进程共同修改同一数据目录。

**不会持久化原始音频**，因此当前没有录音回放，也不能靠保存记录重新转写原始声音。手动保存主要保存已确认状态，不等同于主动把所有缓冲音频强制识别完成；停止收尾才会处理尾部。

MD 导出含标题、元信息、逐段说话人、起止时间和文字。JSON 导出结构为 `meta` 与 `transcript.paragraphs`，不包含 embedding 和完整恢复状态，因此不能把导出 JSON 当成可直接导入续记的备份。

## 8. HTTP API

所有业务 HTTP 路径以 `/api` 开头。

| 方法 | 路径 | 作用 / 请求要点 |
| --- | --- | --- |
| GET | /api/health | 配置、模型加载状态、活动会话数 |
| GET | /api/meetings | 列表，返回 meetings |
| POST | /api/meetings | 创建，JSON `{title}`，返回 meeting |
| GET | /api/meetings/{id} | 标题和参与者、热词、阈值配置 |
| GET | /api/meetings/{id}/segments | 片段摘要，不返回声纹 |
| POST | /api/meetings/{id}/save | 保存活动会议，或返回已有元信息 |
| DELETE | /api/meetings/{id} | 删除会议目录 |
| GET | /api/meetings/{id}/export?format=md或json | 下载记录，默认 md |
| GET | /api/speakers | 返回 speakers，包含向量元信息 |
| POST | /api/speakers | 创建，JSON `{name}` |
| PATCH | /api/speakers/{id} | 改名，JSON `{name}` |
| DELETE | /api/speakers/{id} | 删除说话人及声纹 |
| POST | /api/speakers/{id}/vectors/record | application/octet-stream 原始 PCM |
| POST | /api/speakers/{id}/vectors/meeting | JSON `{meeting_id, segment_ids}` |
| DELETE | /api/speakers/{id}/vectors/{vector_id} | 删除单条声纹 |
| GET | /api/hotwords | 返回 hotwords |
| POST | /api/hotwords | JSON `{name, words: string[]}` |
| PATCH | /api/hotwords/{id} | 可更新 name、words |
| DELETE | /api/hotwords/{id} | 删除热词库 |

会议详情 GET 主要供配置初始化使用，不返回完整转写；当前页面通过 WebSocket 恢复转写。常规 API 失败时前端读取 JSON 的 detail 展示错误。

## 9. WebSocket 协议

端点：`/ws/audio`。

查询参数有 `session_id`、`meeting_id`、`title`、`vad_threshold`、`rec_threshold`、`participants`、`hotwords`、`spk_threshold`。participants 和 hotwords 是逗号分隔的 ID 字符串。

后端把处理阈值限制为 0.5～30 秒、确认阈值限制为 0.2～30 秒、声纹匹配阈值限制为 0～1。配置参数未提供时可以恢复会议旧值；显式传空串表示清空选择。未提供 meeting_id 的连接可以识别，但没有可供会议持久化使用的 ID。

客户端发送：二进制 PCM 块，以及收尾控制 JSON `{"type":"stop"}`。

服务端消息：

| type | 主要字段 | 含义 |
| --- | --- | --- |
| ready | session_id、sample_rate、阈值、meeting_id、resumed、参与者和热词配置 | 初始化完成 |
| transcript | version、segments | 完整可信片段列表，前端整体替换 |
| state | buffered_ms | 当前未裁剪缓冲时长 |
| closed | 无额外必要字段 | 后端收尾完成 |
| error | message | 处理器崩溃时可能发送 |

transcript 示例（演示数据）：

```json
{"type":"transcript","version":1,"segments":[{"id":1,"speaker_id":1,"speaker_name":"说话人1","start_ms":0,"end_ms":3200,"text":"我们开始讨论今天的安排。"}]}
```

前端 `ServerMessage` 联合类型和 switch 当前没有处理后端 `error` 消息；讨论错误处理改造时需要同时检查两端协议。

## 10. 配置

后端读取进程环境变量，没有看到自动加载 `.env` 的实现。

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| VAD_THRESHOLD_S | 10.0 | 缓冲处理触发时长 |
| REC_THRESHOLD_S | 2.0 | 句尾确认安全间隔 |
| MERGE_GAP_S | 2.0 | 展示/导出的段落合并间隔 |
| FINAL_DISCARD_S | 1.0 | 停止时不足此时长的尾部不识别 |
| MEETINGS_DIR | meetings | 会议目录 |
| SPEAKERS_DIR | speakers | 声纹目录 |
| HOTWORDS_DIR | hotwords | 热词目录 |
| CLUSTER_MERGE_THR | 0.78 | 聚类合并阈值 |
| MIN_SPEAKER_AUDIO_S | 5.0 | 短声纹提示阈值 |
| SPK_MATCH_THRESHOLD | 0.3 | 参与者匹配阈值 |
| ASR_MODEL | paraformer-zh | ASR 名称或本地目录 |
| FSMN_VAD_MODEL | fsmn-vad | VAD 名称或本地目录 |
| PUNC_MODEL | ct-punc-c | 标点模型名称或本地目录 |
| SPK_MODEL | cam++ | 声纹模型名称或本地目录 |
| MODELSCOPE_CACHE | 用户目录/.cache/modelscope/hub/models | 本项目查找模型仓库目录的根路径 |

`resolve_model()` 先接受存在的本地路径，再将短名映射为仓库，检查 `MODELSCOPE_CACHE/<仓库路径>`，找不到则把仓库标识交给 FunASR。不能仅凭项目中存在 models_cache 就认为程序一定会使用它，需要核对目录层级和环境变量。

前端 `VITE_WS_URL` 可覆盖 WebSocket **完整端点**，例如 `ws://127.0.0.1:8000/ws/audio`。没有配置时使用页面同源 `/ws/audio`。HTTP BASE 固定为 `/api`，并没有对应的 HTTP BASE 环境变量。

会议页本身初始处理/确认阈值写为 10 / 2，匹配阈值也会从页面配置传入；因此后端环境默认值不一定就是页面连接实际使用的值。

## 11. 本地启动说明

以下是按源码整理的命令，未在本次文档工作中执行。应在已正确配置的 Python 环境及 Node 环境中运行。前端声明 Node 版本为 `^22.18.0 || >=24.12.0`。

后端 PowerShell 终端：

```powershell
Set-Location E:\Personal_Project\meeting
uv sync
# 只有已将模型缓存迁移到根目录时才设置以下路径，请核对内部 iic/... 层级。
$env:MODELSCOPE_CACHE = Join-Path (Get-Location) 'models_cache\models'
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

如果 uv 未加入 PATH，请先安装 uv 或指定本机已有的 uv 可执行文件。旧目录里的 uv-bin 不属于仓库管理文件，Git 不会自动搬迁。模型缓存不完整时仍可能下载模型；“本地推理”不等于首次启动无需网络。

前端另开终端：

```powershell
Set-Location E:\Personal_Project\meeting
pnpm install
pnpm dev
```

浏览器打开 Vite 实际输出的本地地址。开发配置将 `/api` 和 `/ws` 代理到 `http://localhost:8000`，其中 `/ws` 支持 WebSocket 升级。

后端健康地址是 `http://127.0.0.1:8000/api/health`；默认 FastAPI API 文档地址为 `/docs`。麦克风采集依赖浏览器权限及可用的安全上下文，远程部署时应处理 HTTPS / WSS。

前端已有检查命令：`pnpm type-check`、`pnpm build-only`、`pnpm build`；build 会执行类型检查与生产构建。

开发代理不会自动变成生产部署配置。正式部署需要安排静态文件服务、Vue history 路由回退、`/api` 转发和 WebSocket 升级转发；当前没有看到完整部署脚本。

### 11.1 从旧目录迁移

本次仅将 Git 管理的源码从 `backend/backend/` 与 `frontend/frontend/` 合并到根目录。现有本地 `meetings/`、`speakers/`、`hotwords/`、`models_cache/`、`.venv/`、`node_modules/`、`uv-bin/` 均被忽略，不会自动搬迁。迁移前停止旧服务，分别移动要保留的数据至新根目录，避免覆盖既有数据，然后在根目录重新运行 `uv sync` 和 `pnpm install`。

### 11.2 后端模块拆分

原有单一 `app/` 后端包已拆成 `api/`、`core/`、`audio/`、`inference/`、`sessions/`、`storage/` 六组模块，根目录的 `main.py` 负责 FastAPI 注册和预加载。HTTP API、WebSocket、消息格式及原有 pickle 存储保持不变。启动时使用 `uvicorn main:app`，`example/test_stt.py` 使用 `core.config`。

新增 `tests/test_backend_contract.py`，用于检查路由注册及会议存储。没有在本次重构中更改识别算法，完整音频推理、保存续记和多会话并发仍需运行环境验收。

## 12. 测试资料与验证边界

`example/stream_test.py` 读取示例 WAV、转单声道、重采样、拼接语音与静音，通过 WebSocket 推送并打印片段/段落，属于手动集成验证工具；它默认不带 meeting_id，因此不验证会议保存流程。其发送节奏也不是严格按真实音频时长播放。

`example/test_stt.py` 用 FunASR 直接处理示例音频，属于模型可用性演示。其配置导入已更正为 `from core.config import resolve_model`，应从仓库根目录运行。

这些脚本不等于系统性的自动化测试。已增加基本接口和存储测试，但尚缺完整保存续记、并发与语音推理自动化测试。本次没有验证现有日志是否仍能代表当前运行状态。

## 13. 当前限制与值得核查的实现问题

以下由源码静态分析得出，不表示已复现故障，也不表示已修复：

1. **识别与声纹提取绑定**：某个句子若无法得到 embedding，当前会跳过该片段，即便 ASR 已识别出文字，也不会单独保留纯文本。
2. **保存结果反馈不够严格**：`_do_save()` 捕获异常只记日志；保存完成事件 `_save_done` 设置后没有在每次请求前明确清除，后续保存请求可能提前返回。导出端也未严格检查 request_save 的布尔结果。不能把所有成功提示等同于已确认最新状态落盘。
3. **重复连接与同会话操作**：新会议连接停止旧处理器时未先等待其完整收尾再读取最终状态；同一会议多标签页、反复应用设置的并发行为需要验证。前端重新创建 RealtimeClient 的流程也需检查旧连接是否被完整释放。
4. **处理长会话的成本**：可信段保存在内存，全量匹配、全量聚类、完整快照会随会议增长变重；纯静音或长期未确认的缓冲没有看到硬性时长上限。
5. **并发容量**：全局推理锁使多个会议共享串行推理吞吐。进程内会话字典及 pickle 存储不适合直接扩成多 worker 共享写入。
6. **错误反馈**：后端 error 消息未进入前端协议处理；模型或最终识别异常时，用户看到的错误可能不充分。
7. **生产访问控制**：当前 CORS 允许任意源，没有鉴权、权限隔离和业务级限流，不应将其描述为已具备多租户生产能力。
8. **数据格式边界**：pickle 只适用于可信本地数据，不能直接把任意外部 pickle 当作导入文件。会议和声纹包含敏感业务数据，原始录音不落盘并不代表没有敏感数据。

目前没有看到已实现的音频文件上传转写界面、录音回放、人工修改转写文本、人工纠正片段说话人、智能摘要、待办提取、全文搜索、JSON 导入恢复、云同步、账号权限、系统音频采集或会议软件接入。这些可以作为未来需求，不能当成现有能力。

## 14. 后续开发从哪里改

| 需求 | 优先查看 |
| --- | --- |
| 转写延迟、尾句遗漏、窗口增长 | processor.py、buffer.py、config.py |
| 识别模型、标点、声纹、推理设备 | models.py、config.py |
| 姓名误匹配、未知说话人标签抖动 | processor.py 的 _match_participant / _recluster / _remap_speakers |
| 保存、续记、时长、数据迁移 | processor.py 的持久化方法、store.py、main.py |
| 声纹录制与会议片段导入 | SpeakerVectorAdder.vue、speaker_store.py、main.py |
| 热词效果、解析、数量限制 | hotword_store.py、processor.py、HotwordsPage.vue |
| 麦克风无声、重采样、音频传输 | recorder.ts、pcm-processor.ts、api/ws.ts |
| 页面交互和段落显示 | MeetingPage.vue、HomePage.vue、assets/main.css |
| 新增接口和消息类型 | api/websocket.py、api/client.ts、api/ws.ts、types.ts |
| 摘要、待办或 GPT 接入 | 当前没有实现；可基于导出的文字设计独立后处理流程 |

修改时应保留这些约束：已确认文本与声纹不因重聚类而重复计算；前后端的时间单位与段落合并规则一致；完整快照不能误当追加消息；姓名匹配和未知聚类是两条分支；没有原始录音时无法补做依赖原音频的处理。

## 15. 提供给 GPT 的开场说明

可以在附带本文件时使用下面这段话：

> 这是我的 Meeting of Pioneers 项目上下文，请先阅读整个文件，再协助我讨论或开发。项目是 Vue 3 + TypeScript 前端、FastAPI + FunASR 后端的本地中文会议转写应用，包含实时麦克风上传、可信句段确认、声纹匹配与未知说话人聚类、会议保存续记、声纹库和热词库。请区分已有实现、静态分析发现的待验证问题，以及你提出的新方案。不要假设项目已经有智能摘要、数据库、用户账号或原始录音保存。如果你没有最新源码，请明确哪些判断还需要查看具体文件，不要声称已读取我的本地项目或已完成代码修改。

本文是截至整理日期的项目快照，不包含全部源码。后续功能、协议或目录变化后，应同步更新相应章节。
