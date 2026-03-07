#!/bin/bash
# Kimi Code CLI 启动脚本 - 使用项目自定义 Agent 配置
# 用法: ./kimi.sh [额外参数]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_FILE="$SCRIPT_DIR/.kimi/agent.yaml"

if [ ! -f "$AGENT_FILE" ]; then
    echo "Error: Agent 配置文件不存在: $AGENT_FILE"
    exit 1
fi

echo "🚀 启动 Kimi Code CLI 使用项目自定义 Agent 配置..."
echo "📄 Agent 配置: $AGENT_FILE"
echo ""

kimi --agent-file "$AGENT_FILE" "$@"
