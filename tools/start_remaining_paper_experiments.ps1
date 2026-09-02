$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$base = Join-Path $root 'outputs/paper_evidence_20260720'
$logs = Join-Path $base 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$state = Join-Path $base 'remaining_queue_state.jsonl'

function Run-Step([string]$Name, [string[]]$CommandArgs) {
    @{time=(Get-Date).ToString('o');step=$Name;status='started'} | ConvertTo-Json -Compress | Add-Content $state
    & python @CommandArgs *> (Join-Path $logs "$Name.log")
    if ($LASTEXITCODE -ne 0) { throw "$Name failed: $LASTEXITCODE" }
    @{time=(Get-Date).ToString('o');step=$Name;status='completed'} | ConvertTo-Json -Compress | Add-Content $state
}

try {
    Run-Step 'grandstaff_relation_ablation_200' @('tools/run_cross_dataset_relation_edge_ablation.py','--dataset','grandstaff','--out','outputs/paper_evidence_20260720/relation_edge_ablation_grandstaff','--limit','200','--bootstrap-samples','10000')
    Run-Step 'olimpic_relation_ablation_200' @('tools/run_cross_dataset_relation_edge_ablation.py','--dataset','olimpic','--out','outputs/paper_evidence_20260720/relation_edge_ablation_olimpic','--limit','200','--bootstrap-samples','10000')
    Run-Step 'cross_dataset_tests' @('-m','pytest','tests/test_run_cross_dataset_relation_edge_ablation.py','-q')
    @{time=(Get-Date).ToString('o');status='all_completed'} | ConvertTo-Json -Compress | Add-Content $state
} catch {
    @{time=(Get-Date).ToString('o');status='failed';error=$_.Exception.Message} | ConvertTo-Json -Compress | Add-Content $state
    exit 1
}
