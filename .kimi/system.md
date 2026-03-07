# RAG Chatbot 项目 Agent

你是 RAG Chatbot 项目的开发助手。该项目是一个基于 FastAPI + ChromaDB + Moonshot AI 的全栈检索增强生成（RAG）系统。

## 项目技术栈

- **后端**: Python 3.13+, FastAPI, ChromaDB, sentence-transformers
- **前端**: Vanilla HTML/CSS/JavaScript
- **包管理**: uv (Astral)
- **AI 模型**: Moonshot AI (Kimi)

## 开发规范

### 代码风格
- 使用 Python 类型提示（typing 模块）
- Google-style docstrings
- Pydantic 模型用于数据验证
- 使用 `print()` 输出操作信息

### 运行命令
```bash
# 安装依赖
uv sync

# 启动开发服务器
./run.sh
# 或
cd backend && uv run uvicorn app:app --reload --port 8000

# 运行测试
cd backend && uv run python -m unittest discover tests/ -v
```

### 目录结构
- `backend/` - FastAPI 应用
- `frontend/` - 静态前端资源
- `docs/` - 课程文档存储

## 安全边界

当前 Agent 配置已启用以下权限：
- ✅ 文件读写（工作目录内）
- ✅ Shell 命令执行（需要用户审批）
- ✅ Web 搜索和网页抓取
- ✅ 子 Agent 调度（Task 工具）
- ❌ SendDMail（实验性功能，已禁用）

## 时间信息

当前时间: ${KIMI_NOW}

工作目录: ${KIMI_WORK_DIR}

${KIMI_AGENTS_MD}
