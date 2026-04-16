@echo off
REM Internal: load vcvars64 then print env. Consumed by bolt-env.sh.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo [error] vcvars64 failed 1>&2
    exit /b 1
)
set
