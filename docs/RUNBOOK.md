# CADx Loop Engine Runbook

This document covers operational tasks, process monitoring, recovery, and diagnostic steps for the self-sustaining Prostate Cancer CADx daemon.

## Process Layout

The system consists of three processes:
1. **Daemon (`backend/daemon.py`)**: Executes pending skills from SQLite `db/cadx.db` and writes run statistics.
2. **Watchdog (`backend/watchdog.py`)**: Checks every 15 seconds if `daemon.py` is running and restarts it if it crashed.
3. **Supervisor API (`backend/api.py`)**: Exposes status, metrics, and GPU metrics on `http://127.0.0.1:8600`.

## Process Management Commands

### Manual Start / Recovery
If you need to manually restart the watchdog:
```powershell
.\venv\Scripts\python.exe backend/watchdog.py
```
This will automatically spawn `backend/daemon.py` in the background.

### Checking Process Status
Verify that the daemon and watchdog are active:
```powershell
Get-Process -Name python | Select-Object Id, CPU, WorkingSet
```
Or check the API endpoint:
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8600/status
```

### Resetting database/state
If the DB becomes corrupted or you wish to re-run from scratch, delete the DB file and seed it:
```powershell
Remove-Item db/cadx.db
.\venv\Scripts\python.exe scripts/wire.py
```
This recreates the SQLite tables and seeds the 100 skills in a pending state.
