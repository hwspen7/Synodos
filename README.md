# Meeting of Pioneers

本地中文会议转写应用，使用 Vue 3 + TypeScript 前端与 FastAPI + FunASR 后端。

支持麦克风实时上传、语音转写与标点恢复、说话人聚类、参与者声纹匹配、热词库、会议保存续记，以及 Markdown / JSON 导出。

完整架构、接口、配置和使用说明见 [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)。

## 本地开发

后端（Python ≥3.12，已安装 uv）：

```powershell
cd backend/backend
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

模型首次运行可能需要下载；也可以通过 `ASR_MODEL`、`FSMN_VAD_MODEL`、`PUNC_MODEL`、`SPK_MODEL` 指定本地模型目录，详见完整说明。

前端（Node `^22.18.0 || >=24.12.0`，已安装 pnpm），另开终端：

```powershell
cd frontend/frontend
pnpm install
pnpm dev
```

打开 Vite 输出的地址并允许麦克风权限。开发代理将 `/api`、`/ws` 转发到后端 8000 端口。

## 仓库内容

仓库只保存源码、配置、依赖锁文件和文档。虚拟环境、node_modules、模型缓存、工具安装包、日志、示例音频，以及本地会议、声纹和热词数据均被 `.gitignore` 排除，继续保留在本机。

运行时自动创建默认数据目录。项目不保存原始会议录音；导出的 JSON 不包含完整续记状态。

`backend/backend/example/stream_test.py` 默认引用的示例 WAV 未纳入仓库，运行时请通过 `--wav` 指定自己的音频文件。
