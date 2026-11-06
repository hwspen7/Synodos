# Execute the actual Phase 1/2 checker with an artificial native process.
# Proves harmless stderr cannot become a PowerShell NativeCommandError, and
# proves a real failing exit status remains a failure.
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runner = Join-Path $repo "tools\check_integrated_integrity.ps1"
$work = Join-Path $env:TEMP ("phase12-stderr-check-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $work -Force | Out-Null
$original = [Environment]::GetEnvironmentVariable("MEETING_PHASE12_SMOKE_EXIT", "Process")
try {
    $exe = Join-Path $work "NativeUnittestFixture.exe"
    $source = @'
using System;
public class NativeUnittestFixture {
    public static int Main(string[] args) {
        Console.Error.WriteLine("Rejecting non-conserving speaker split for segment 1");
        Console.Out.WriteLine("Synthetic unittest diagnostics");
        string desired = Environment.GetEnvironmentVariable("MEETING_PHASE12_SMOKE_EXIT");
        return desired == "7" ? 7 : 0;
    }
}
'@
    Add-Type -TypeDefinition $source -OutputAssembly $exe `
        -OutputType ConsoleApplication -ErrorAction Stop

    foreach ($case in @(
        @{ Exit = "0"; Expected = 0; Marker = "Result: PASS" },
        @{ Exit = "7"; Expected = 1; Marker = "Result: FAIL" }
    )) {
        $env:MEETING_PHASE12_SMOKE_EXIT = $case.Exit
        $stdout = Join-Path $work ("runner-" + $case.Exit + ".stdout")
        $stderr = Join-Path $work ("runner-" + $case.Exit + ".stderr")
        $process = Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
            -ArgumentList @(
                "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                ('"' + $runner + '"'), "-PythonExe", ('"' + $exe + '"')
            ) -Wait -PassThru -NoNewWindow `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $output = (Get-Content -LiteralPath $stdout -Raw) + "`n" +
            (Get-Content -LiteralPath $stderr -Raw)
        if ($process.ExitCode -ne $case.Expected) {
            throw "Expected nested PowerShell exit $($case.Expected), got $($process.ExitCode): $output"
        }
        if (-not $output.Contains($case.Marker)) {
            throw "Missing expected status marker $($case.Marker): $output"
        }
        if ($output.Contains("NativeCommandError")) {
            throw "Native stderr was incorrectly promoted to a PowerShell error"
        }
        Write-Output ("Phase 1/2 native stderr + exit-code smoke PASS (child exit " +
            $case.Exit + ")")
    }
} finally {
    [Environment]::SetEnvironmentVariable("MEETING_PHASE12_SMOKE_EXIT", $original, "Process")
    Remove-Item -LiteralPath $work -Force -Recurse -ErrorAction SilentlyContinue
}
