param(
    [string]$SourceRoot = "",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$workspaceRoot = (Get-Item (Join-Path $PSScriptRoot "..\..\..\..\..")).FullName
if (-not $SourceRoot) {
    $SourceRoot = Join-Path $workspaceRoot "data\carla_self_collection\production_33k_v1_0\emergency"
}
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $workspaceRoot "data\carla_self_collection\deliverables\emergency_train_preview_v1_20260726"
}

$categoryOrder = @(
    "E1_vehicle_cut_in",
    "E2_critical_pedestrian",
    "E2_hard_pedestrian",
    "E2_safe_pedestrian",
    "E4_lead_hard_brake"
)

if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) {
    throw "Source root does not exist: $SourceRoot"
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

$validEpisodes = [System.Collections.Generic.List[object]]::new()
$summaryRows = [System.Collections.Generic.List[object]]::new()

foreach ($category in $categoryOrder) {
    $categoryRoot = Join-Path $SourceRoot $category
    if (-not (Test-Path -LiteralPath $categoryRoot -PathType Container)) {
        throw "Missing category directory: $categoryRoot"
    }

    $categoryEpisodes = Get-ChildItem -LiteralPath $categoryRoot -Directory -Recurse |
        Where-Object { $_.Name -like "episode_*" } |
        Sort-Object FullName

    $usableCount = 0
    $rejectedCount = 0

    foreach ($episode in $categoryEpisodes) {
        $annotationDir = Join-Path $episode.FullName "annotations"
        $annotationFiles = @(
            Get-ChildItem -LiteralPath $annotationDir -Filter "frame_*.json" -File -ErrorAction SilentlyContinue |
                Sort-Object Name
        )

        $allValid = $annotationFiles.Count -eq 10
        $firstAnnotation = $null

        foreach ($annotationFile in $annotationFiles) {
            $annotation = Get-Content -LiteralPath $annotationFile.FullName -Raw -Encoding UTF8 |
                ConvertFrom-Json
            if ($null -eq $firstAnnotation) {
                $firstAnnotation = $annotation
            }
            if ($annotation.sample_valid -ne $true) {
                $allValid = $false
            }
        }

        if (-not $allValid) {
            $rejectedCount += 1
            continue
        }

        $relativeEpisode = $episode.FullName.Substring($SourceRoot.Length).TrimStart("\")
        $mapWeather = Split-Path -Leaf (Split-Path -Parent $episode.FullName)
        $episodeFiles = @(Get-ChildItem -LiteralPath $episode.FullName -File -Recurse)
        $episodeBytes = ($episodeFiles | Measure-Object -Property Length -Sum).Sum

        $validEpisodes.Add([pscustomobject]@{
            category = $category
            map_weather = $mapWeather
            town = [string]$firstAnnotation.town
            weather = [string]$firstAnnotation.weather
            episode = $episode.Name
            relative_path = $relativeEpisode.Replace("\", "/")
            frames = 10
            bytes = [int64]$episodeBytes
            absolute_path = $episode.FullName
        })
        $usableCount += 1
    }

    $categoryRows = @($validEpisodes | Where-Object category -eq $category)
    $categoryBytes = ($categoryRows | Measure-Object -Property bytes -Sum).Sum
    $summaryRows.Add([pscustomobject]@{
        category = $category
        valid_episodes = $usableCount
        valid_frames = $usableCount * 10
        rejected_episode_directories = $rejectedCount
        valid_bytes = [int64]$categoryBytes
        validation = "10/10 sample_valid=true"
    })
}

$indexPath = Join-Path $OutputRoot "dataset_index.tsv"
$validEpisodes |
    Select-Object category, map_weather, town, weather, episode, relative_path, frames, bytes |
    Export-Csv -LiteralPath $indexPath -Delimiter "`t" -NoTypeInformation -Encoding UTF8

$summaryPath = Join-Path $OutputRoot "validation_summary.tsv"
$summaryRows |
    Export-Csv -LiteralPath $summaryPath -Delimiter "`t" -NoTypeInformation -Encoding UTF8

$totalEpisodes = ($summaryRows | Measure-Object -Property valid_episodes -Sum).Sum
$totalFrames = ($summaryRows | Measure-Object -Property valid_frames -Sum).Sum
$totalBytes = ($summaryRows | Measure-Object -Property valid_bytes -Sum).Sum

$readme = @'
# CARLA Emergency Train Preview v1

This delivery was strictly filtered from `production_33k_v1_0/emergency`.

## Dataset size

- Categories: __TOTAL_CATEGORIES__
- Valid episodes: __TOTAL_EPISODES__
- Valid frames: __TOTAL_FRAMES__
- Frames per episode: 10
- Sampling frequency: 2 Hz (0.5 seconds between adjacent samples)
- CARLA: 0.9.15
- Annotation format: schema-v1.1
- Selection rule: exactly 10 `frame_*.json` files and `sample_valid=true` in every frame

## Categories

- E1_vehicle_cut_in: adjacent vehicle cut-in
- E2_critical_pedestrian: critical-distance pedestrian crossing
- E2_hard_pedestrian: hard pedestrian crossing
- E2_safe_pedestrian: safe-distance pedestrian crossing
- E4_lead_hard_brake: lead vehicle hard braking

E3_construction_merge is not included because the production directory does
not yet contain a complete 10/10 valid episode.

## Episode structure

- `annotations/`: per-frame JSON annotations
- `sensors/`: six cameras, LiDAR, and other sensor files
- `bev/`: bird's-eye-view images
- `calib/`: sensor calibration
- `episode_manifest.json`: episode metadata

## Delivery metadata

- `dataset_index.tsv`: category, map, weather, frame count, path, and bytes
- `validation_summary.tsv`: accepted and rejected counts by category
- `SHA256SUMS.txt`: SHA-256 hashes for archives and metadata

## Intended use

This preview is intended for DataLoader integration, training-pipeline
validation, and small overfitting tests. It is not the final 12,100-frame
emergency dataset: category counts are not balanced, E1 is small, and E3 is
missing.
'@
$readme = $readme.Replace("__TOTAL_CATEGORIES__", [string]$categoryOrder.Count)
$readme = $readme.Replace("__TOTAL_EPISODES__", [string]$totalEpisodes)
$readme = $readme.Replace("__TOTAL_FRAMES__", [string]$totalFrames)
Set-Content -LiteralPath (Join-Path $OutputRoot "README.md") -Value $readme -Encoding UTF8

foreach ($category in $categoryOrder) {
    $archivePath = Join-Path $OutputRoot "$category.zip"
    $partialPath = "$archivePath.partial"
    if ((Test-Path -LiteralPath $archivePath) -or (Test-Path -LiteralPath $partialPath)) {
        throw "Archive target already exists; refusing to overwrite: $archivePath"
    }

    $rows = @($validEpisodes | Where-Object category -eq $category)
    Write-Host "Packaging $category`: $($rows.Count) episodes / $($rows.Count * 10) frames"

    $stream = [System.IO.File]::Open(
        $partialPath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
    try {
        $archive = [System.IO.Compression.ZipArchive]::new(
            $stream,
            [System.IO.Compression.ZipArchiveMode]::Create,
            $false
        )
        try {
            foreach ($row in $rows) {
                $episodeRoot = [string]$row.absolute_path
                foreach ($file in Get-ChildItem -LiteralPath $episodeRoot -File -Recurse) {
                    $relativeWithinEpisode = $file.FullName.Substring($episodeRoot.Length).TrimStart("\")
                    $entryName = (
                        "$category/$($row.map_weather)/$($row.episode)/$relativeWithinEpisode"
                    ).Replace("\", "/")
                    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                        $archive,
                        $file.FullName,
                        $entryName,
                        [System.IO.Compression.CompressionLevel]::Fastest
                    ) | Out-Null
                }
            }
        }
        finally {
            $archive.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }

    Move-Item -LiteralPath $partialPath -Destination $archivePath
}

$checksumTargets = @(
    Get-ChildItem -LiteralPath $OutputRoot -File |
        Where-Object { $_.Name -ne "SHA256SUMS.txt" } |
        Sort-Object Name
)
$checksumLines = foreach ($file in $checksumTargets) {
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $($file.Name)"
}
Set-Content -LiteralPath (Join-Path $OutputRoot "SHA256SUMS.txt") -Value $checksumLines -Encoding ASCII

Write-Host ""
Write-Host "Emergency preview package complete."
Write-Host "Episodes : $totalEpisodes"
Write-Host "Frames   : $totalFrames"
Write-Host "Data GB  : $([math]::Round($totalBytes / 1GB, 2))"
Write-Host "Output   : $OutputRoot"
