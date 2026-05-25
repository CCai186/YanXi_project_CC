@echo off
echo ===========================================
echo  AI 语音通话助手 — 环境安装
echo ===========================================
echo.

echo [1/4] 安装 Python 依赖...
pip install -r requirements.txt
if errorlevel 1 (
    echo 依赖安装失败，请检查 Python 和 pip 是否已安装
    pause
    exit /b 1
)

echo.
echo [2/4] 构建知识库...
python -m src.knowledge.dataset_loader --sample
python -m src.knowledge.embedder

echo.
echo [3/4] 配置 API Key
echo 请在 config.yaml 中填写你的 DeepSeek API Key
echo 获取地址: https://platform.deepseek.com/
echo.

echo [4/4] 启动！
echo 运行方式:
echo   python src/main.py          语音模式
echo   python src/main.py --text   文本模式
echo   python src/main.py --test   测试模式
echo.
echo ===========================================
echo  安装完成！
echo ===========================================
pause
