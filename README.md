# Meeting of Pioneers

本地中文会议转写应用，使用 Vue 3 + TypeScript 前端与 FastAPI + FunASR 后端。

支持麦克风实时上传、语音转写与标点恢复、说话人聚类、参与者声纹匹配、热词库、会议保存续记，以及 Markdown / JSON 导出。

完整架构、接口、配置和使用说明见 [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)。

## 统一根目录结构

```text
meeting/
├─ main.py              FastAPI 启动入口
├─ api/                 HTTP / WebSocket 接口
├─ core/                全局配置与共享状态
├─ audio/               后端音频缓冲
├─ inference/           ASR、声纹和聚类模型
├─ sessions/            会话和转写处理
├─ storage/             会议、声纹和热词存储
├─ tests/               后端回归测试
├─ example/             后端手动测试
├─ src/                 前端 Vue 源码
├─ public/              前端静态资源
├─ index.html
├─ vite.config.ts
├─ package.json
├─ pnpm-lock.yaml
├─ pyproject.toml
├─ uv.lock
├─ README.md
└─ PROJECT_CONTEXT.md
```

前后端共享同一个工作目录，无需再进入 `backend` 或 `frontend` 子目录。

## 本地开发

后端（Python ≥3.12，已安装 uv）：

```powershell
uv sync
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

模型首次运行可能需要下载；也可以通过 `ASR_MODEL`、`FSMN_VAD_MODEL`、`PUNC_MODEL`、`SPK_MODEL` 指定本地模型目录，详见完整说明。

前端（Node `^22.18.0 || >=24.12.0`，已安装 pnpm），另开终端：

```powershell
pnpm install
pnpm dev
```

打开 Vite 输出的地址并允许麦克风权限。开发代理将 `/api`、`/ws` 转发到后端 8000 端口。

## 从旧目录迁移

本次将原 `backend/backend/`、`frontend/frontend/` 中的源码合并到仓库根目录。若本机已保存会议和声纹等业务数据，须在停止旧服务后，手动把原 `backend/backend/meetings/`、`speakers/`、`hotwords/` 迁移到根目录下的同名目录。模型缓存也需按实际存放位置迁移或重新配置 `MODELSCOPE_CACHE`，不要覆盖现有数据。Git 不会自动搬迁被忽略的本地文件。

## 仓库内容

仓库只保存源码、配置、依赖锁文件和文档。虚拟环境、node_modules、模型缓存、工具安装包、日志、示例音频，以及本地会议、声纹和热词数据均被 `.gitignore` 排除，继续保留在本机。

运行时自动创建默认数据目录。项目不保存原始会议录音；导出的 JSON 不包含完整续记状态。

`example/stream_test.py` 默认引用的示例 WAV 未纳入仓库，运行时请通过 `--wav` 指定自己的音频文件。

## 后端模块划分

`main.py` 只组装 FastAPI、路由和模型预加载。业务 API 放在 `api/`，共享实例放在 `core/`，音频缓冲在 `audio/`，模型调用在 `inference/`，实时会话和处理线程在 `sessions/`，会议、声纹与热词持久化在 `storage/`。原有 HTTP、WebSocket 路径和协议不变。

后端基础测试：

```powershell
uv run python -m unittest discover -s tests
```

测试不执行语音模型推理。实际模型运行、并发和长会议仍需要额外验收。
