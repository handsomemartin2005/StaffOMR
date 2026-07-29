param([switch]$Force)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$out = Join-Path $root 'outputs/paper_evidence_20260720'
$logs = Join-Path $out 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$state = Join-Path $out 'queue_state.jsonl'

function Run-Step([string]$Name, [string[]]$Command) {
    $done = Join-Path $logs "$Name.done"
    if ((Test-Path $done) -and -not $Force) { return }
    @{time=(Get-Date).ToString('o'); step=$Name; status='started'} | ConvertTo-Json -Compress | Add-Content $state
    & $Command[0] $Command[1..($Command.Count-1)] *> (Join-Path $logs "$Name.log")
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
    New-Item -ItemType File -Force -Path $done | Out-Null
    @{time=(Get-Date).ToString('o'); step=$Name; status='completed'} | ConvertTo-Json -Compress | Add-Content $state
}

try {
    Run-Step 'error_decomposition' @('python','tools/build_transfer_error_decomposition.py','--dataset','debussy','--out','outputs/paper_evidence_20260720/error_decomposition')
    Run-Step 'debussy_relation_ablation' @('python','tools/run_relation_edge_ablation.py','--out','outputs/paper_evidence_20260720/relation_edge_ablation','--limit','24','--bootstrap-samples','10000')
    Run-Step 'oracle_diagnostics' @('python','tools/build_oracle_diagnostics.py','--dataset','debussy','--out','outputs/paper_evidence_20260720/oracle_diagnostics')
    Run-Step 'figures' @('python','tools/build_paper_evidence_figures.py','--out','outputs/paper_evidence_20260720/figures')
    Run-Step 'focused_tests' @('python','-m','pytest','tests/test_paper_evidence_metrics.py','tests/test_build_transfer_error_decomposition.py','tests/test_run_relation_edge_ablation.py','tests/test_paper_evidence_oracles.py','tests/test_build_paper_evidence_figures.py','-q')
    @{time=(Get-Date).ToString('o'); status='all_completed'} | ConvertTo-Json -Compress | Add-Content $state
} catch {
    @{time=(Get-Date).ToString('o'); status='failed'; error=$_.Exception.Message} | ConvertTo-Json -Compress | Add-Content $state
    exit 1
}
