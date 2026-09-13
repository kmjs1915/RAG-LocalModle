<#
RAG-LocalEmbedding Quick Start Script
Save at E:\RAG-LocalEmbedding\Start-RAG.ps1
FastAPI 版：入口为 main.py（原 Gradio 版 app.py 已移除），监听端口 4060
#>
$ProjectRoot = "E:\RAG-LocalEmbedding"
Set-Location $ProjectRoot

# Get LAN IPv4 address, exclude loopback and virtual adapters
$lanIp = (Get-NetIPAddress -AddressFamily IPv4 -Status Preferred | Where-Object {
    $_.IPAddress -notlike "127.*" -and $_.PrefixOrigin -ne "WellKnown" `
    -and $_.InterfaceAlias -notmatch "vEthernet|WSL|Loopback|Tunnel"
} | Select-Object -First 1).IPAddress

Clear-Host
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "        Local RAG KnowledgeBase Starter" -ForegroundColor Cyan
Write-Host "          (FastAPI + 静态前端, 端口 4060)" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Project Folder: $ProjectRoot"
if ($lanIp) {
    Write-Host "✅ Detected LAN IP: $lanIp" -ForegroundColor Green
    Write-Host "👉 Local access: http://127.0.0.1:4060"
    Write-Host "👉 LAN access: http://$lanIp`:4060" -ForegroundColor Yellow
    Write-Host "👉 接口文档: http://127.0.0.1:4060/api/docs" -ForegroundColor DarkCyan
}
else {
    Write-Host "⚠️ No LAN IP found. Only local: http://127.0.0.1:4060" -ForegroundColor DarkYellow
}
Write-Host "=============================================`n" -ForegroundColor Cyan

# Activate venv
$venvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
    Write-Host "✅ Virtual env (.venv) activated`n" -ForegroundColor Green
}
else {
    Write-Host "❌ Cannot find .venv activate script!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit
}

# Start app（FastAPI：python main.py；仅本机访问可加 --host 127.0.0.1）
python main.py
