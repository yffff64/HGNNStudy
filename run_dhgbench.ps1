# ============================================================
#  DHG-Bench 一键运行脚本 (Windows / PowerShell)
#  用法示例:
#    .\run_dhgbench.ps1 -dname cora -task node_cls -method HGNN
#    .\run_dhgbench.ps1 -dname pubmed -task edge_pred -method EDHNN
#    .\run_dhgbench.ps1 -dname stream_player -task hg_cls -method TFHNN
# ============================================================
param(
    [string]$dname = 'cora',
    [string]$task  = 'node_cls',
    [string]$method = 'HGNN',
    [switch]$notDefault
)

# 1) 激活已配置好的 conda 环境
conda activate dhgbench

# 2) 进入 dhgbench 代码目录（数据路径 ../data 基于此目录）
Set-Location "C:\Users\r\Desktop\DHG-Bench-main\DHG-Bench-main\dhgbench"

# 3) 组装并运行命令
$cmd = @("main.py", "--dname=$dname", "--task_type=$task", "--method=$method", "--is_default=$(-not $notDefault)")
Write-Host "Running: python $cmd" -ForegroundColor Cyan
python $cmd
