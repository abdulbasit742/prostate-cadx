# Windows Scheduled Task Installation Script for Prostate Cancer CADx
# Registers the watchdog to start at boot and user logon, ensuring complete autonomy.

$WorkingDir = "C:\Users\absh5\.gemini\antigravity\scratch\prostate-cadx"
$PythonPath = "$WorkingDir\venv\Scripts\python.exe"
$ScriptPath = "$WorkingDir\backend\watchdog.py"

# Define the action (run watchdog script)
$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument $ScriptPath -WorkingDirectory $WorkingDir

# Trigger at boot and logon
$Trigger1 = New-ScheduledTaskTrigger -AtStartup
$Trigger2 = New-ScheduledTaskTrigger -AtLogOn

# Principal to run with highest privileges
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Highest

# Task Settings
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Register Scheduled Task
$TaskName = "ProstateCADxWatchdog"
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger @($Trigger1, $Trigger2) -Principal $Principal -Settings $Settings -Force

Write-Host "Windows Scheduled Task '$TaskName' registered successfully."
Write-Host "Watchdog will launch at startup and logon with WorkingDirectory: $WorkingDir"
