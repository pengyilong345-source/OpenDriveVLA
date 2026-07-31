param(
    [string]$PackageRoot = "",
    [string]$OutputRoot = "",
    [int64]$PartSizeBytes = 943718400
)

$ErrorActionPreference = "Stop"

$workspaceRoot = (Get-Item (Join-Path $PSScriptRoot "..\..\..\..\..")).FullName
if (-not $PackageRoot) {
    $PackageRoot = Join-Path $workspaceRoot "data\carla_self_collection\deliverables\emergency_train_preview_v1_20260726"
}
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $workspaceRoot "data\carla_self_collection\deliverables\emergency_train_preview_v1_20260726_wechat_parts"
}

if (-not (Test-Path -LiteralPath $PackageRoot -PathType Container)) {
    throw "Package root does not exist: $PackageRoot"
}
if (Test-Path -LiteralPath $OutputRoot) {
    $existing = Get-ChildItem -LiteralPath $OutputRoot -Force -ErrorAction SilentlyContinue
    if ($existing) {
        throw "Output directory is not empty; refusing to overwrite: $OutputRoot"
    }
}
else {
    New-Item -ItemType Directory -Path $OutputRoot | Out-Null
}

$buffer = New-Object byte[] (8MB)

foreach ($archive in Get-ChildItem -LiteralPath $PackageRoot -Filter "*.zip" -File | Sort-Object Name) {
    $inputStream = [System.IO.File]::OpenRead($archive.FullName)
    try {
        $partNumber = 1
        while ($inputStream.Position -lt $inputStream.Length) {
            $partName = "{0}.part{1:D3}" -f $archive.Name, $partNumber
            $partPath = Join-Path $OutputRoot $partName
            $outputStream = [System.IO.File]::Open(
                $partPath,
                [System.IO.FileMode]::CreateNew,
                [System.IO.FileAccess]::Write,
                [System.IO.FileShare]::None
            )
            try {
                $written = [int64]0
                while ($written -lt $PartSizeBytes -and $inputStream.Position -lt $inputStream.Length) {
                    $remaining = [Math]::Min(
                        [int64]$buffer.Length,
                        $PartSizeBytes - $written
                    )
                    $read = $inputStream.Read($buffer, 0, [int]$remaining)
                    if ($read -le 0) {
                        break
                    }
                    $outputStream.Write($buffer, 0, $read)
                    $written += $read
                }
            }
            finally {
                $outputStream.Dispose()
            }
            Write-Host "$partName  $([math]::Round($written / 1MB, 1)) MB"
            $partNumber += 1
        }
    }
    finally {
        $inputStream.Dispose()
    }
}

foreach ($metadataName in @(
    "README.md",
    "dataset_index.tsv",
    "validation_summary.tsv",
    "SHA256SUMS.txt"
)) {
    Copy-Item -LiteralPath (Join-Path $PackageRoot $metadataName) -Destination $OutputRoot
}

$mergeScript = @'
$ErrorActionPreference = "Stop"
$sourceRoot = $PSScriptRoot
$outputRoot = Join-Path $sourceRoot "reassembled"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$groups = Get-ChildItem -LiteralPath $sourceRoot -Filter "*.zip.part*" -File |
    Group-Object { $_.Name -replace "\.part\d+$", "" }

$buffer = New-Object byte[] (8MB)
foreach ($group in $groups) {
    $outputPath = Join-Path $outputRoot $group.Name
    if (Test-Path -LiteralPath $outputPath) {
        throw "Refusing to overwrite: $outputPath"
    }
    $outputStream = [System.IO.File]::Open(
        $outputPath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    try {
        foreach ($part in $group.Group | Sort-Object Name) {
            $inputStream = [System.IO.File]::OpenRead($part.FullName)
            try {
                while (($read = $inputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                    $outputStream.Write($buffer, 0, $read)
                }
            }
            finally {
                $inputStream.Dispose()
            }
        }
    }
    finally {
        $outputStream.Dispose()
    }
    Write-Host "Reassembled: $outputPath"
}

$expected = @{}
foreach ($line in Get-Content -LiteralPath (Join-Path $sourceRoot "SHA256SUMS.txt")) {
    $parts = $line -split "  ", 2
    if ($parts.Count -eq 2) {
        $expected[$parts[1]] = $parts[0]
    }
}

foreach ($zip in Get-ChildItem -LiteralPath $outputRoot -Filter "*.zip" -File) {
    $actual = (Get-FileHash -LiteralPath $zip.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected[$zip.Name]) {
        throw "SHA-256 mismatch: $($zip.Name)"
    }
    Write-Host "SHA-256 PASS: $($zip.Name)"
}
'@
Set-Content -LiteralPath (Join-Path $OutputRoot "merge_wechat_parts.ps1") -Value $mergeScript -Encoding UTF8

$partHashes = foreach ($part in Get-ChildItem -LiteralPath $OutputRoot -Filter "*.part*" -File | Sort-Object Name) {
    $hash = (Get-FileHash -LiteralPath $part.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $($part.Name)"
}
Set-Content -LiteralPath (Join-Path $OutputRoot "PART_SHA256SUMS.txt") -Value $partHashes -Encoding ASCII

$instructions = @'
# WeChat split delivery

Every `.partNNN` file is at most 900 MiB.

1. Download every part and keep all files in this directory.
2. Right-click `merge_wechat_parts.ps1` and run with PowerShell, or run:

   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\merge_wechat_parts.ps1

3. The restored ZIP files are written to `reassembled`.
4. The merge script automatically verifies their SHA-256 hashes.

Do not try to extract an individual `.partNNN` file.
'@
Set-Content -LiteralPath (Join-Path $OutputRoot "WECHAT_README.md") -Value $instructions -Encoding UTF8

Write-Host ""
Write-Host "WeChat parts complete."
Write-Host "Output: $OutputRoot"
