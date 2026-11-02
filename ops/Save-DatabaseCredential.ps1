param(
    [string]$CredentialFile = 'E:\Personal_Project\meeting_pg_credential.xml'
)

$ErrorActionPreference = 'Stop'
if (-not $IsWindows -and $PSVersionTable.PSEdition -eq 'Core') {
    throw 'Windows DPAPI credential storage is required.'
}
if (Test-Path -LiteralPath $CredentialFile) {
    throw "Credential file already exists: $CredentialFile. Move it aside manually before replacing it."
}
$parent = Split-Path -Parent $CredentialFile
New-Item -ItemType Directory -Path $parent -Force | Out-Null
$secret = Read-Host 'Enter the password for PostgreSQL role meeting' -AsSecureString
$credential = New-Object System.Management.Automation.PSCredential('meeting', $secret)
$credential | Export-Clixml -LiteralPath $CredentialFile
try {
    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $acl = Get-Acl -LiteralPath $CredentialFile
    $acl.SetAccessRuleProtection($true, $false)
    $access = [System.Security.AccessControl.FileSystemAccessRule]::new(
        $identity.User,
        [System.Security.AccessControl.FileSystemRights]::FullControl,
        [System.Security.AccessControl.AccessControlType]::Allow
    )
    $acl.AddAccessRule($access)
    Set-Acl -LiteralPath $CredentialFile -AclObject $acl
} catch {
    Remove-Item -LiteralPath $CredentialFile -Force -ErrorAction SilentlyContinue
    throw 'Unable to secure the credential file. The file was removed.'
}
Write-Host 'Encrypted credential saved for the current Windows user.' -ForegroundColor Green
Write-Host "Credential file: $CredentialFile"
Write-Host 'The scheduled task must run under this same Windows user account.'
