@echo off
REM bolt.cmd — fast-iteration build driver for Windows.
REM
REM Loads the MSVC environment (vcvars64) once per invocation, then runs
REM cmake+ninja in build/ninja-msvc. Incremental no-op builds drop from
REM ~10s (MSBuild) to under a second (Ninja).
REM
REM Usage:
REM   scripts\bolt.cmd configure        (initial configure)
REM   scripts\bolt.cmd build [target]   (build, optional single-target)
REM   scripts\bolt.cmd test             (ctest)
REM   scripts\bolt.cmd all              (configure + build + test)
REM   scripts\bolt.cmd bench tpch       (run bench_tpch_lite.exe)

setlocal ENABLEEXTENSIONS

set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo [bolt] vcvars64.bat not found at %VCVARS%
    exit /b 1
)

REM vcvars is only needed for the first configure. After that, CMake bakes
REM absolute cl.exe paths into build.ninja, so subsequent builds skip it.
set "BUILD_DIR=%~dp0..\build\ninja-msvc"
if not exist "%BUILD_DIR%\build.ninja" (
    call "%VCVARS%" >nul
    if errorlevel 1 (
        echo [bolt] failed to initialize MSVC environment
        exit /b 1
    )
)

set "CMD=%~1"
shift
set "ARG=%~1"

if "%CMD%"=="" set "CMD=build"

if /I "%CMD%"=="configure" goto :cfg
if /I "%CMD%"=="build"     goto :bld
if /I "%CMD%"=="test"      goto :tst
if /I "%CMD%"=="all"       goto :all
if /I "%CMD%"=="bench"     goto :bench
if /I "%CMD%"=="clean"     goto :cln

echo [bolt] unknown command: %CMD%
echo   use: configure ^| build ^| test ^| all ^| bench ^| clean
exit /b 2

:cfg
cmake --preset ninja-msvc
exit /b %errorlevel%

:bld
if not exist "%BUILD_DIR%\build.ninja" (
    cmake --preset ninja-msvc || exit /b %errorlevel%
)
pushd "%BUILD_DIR%"
if "%ARG%"=="" (
    ninja
) else (
    ninja %ARG%
)
set "RC=%errorlevel%"
popd
exit /b %RC%

:tst
ctest --preset ninja-msvc
exit /b %errorlevel%

:all
if not exist "%BUILD_DIR%\build.ninja" cmake --preset ninja-msvc || exit /b %errorlevel%
cmake --build --preset ninja-msvc || exit /b %errorlevel%
ctest --preset ninja-msvc
exit /b %errorlevel%

:bench
if "%ARG%"=="" (
    "%BUILD_DIR%\benchmarks\bench_bolt.exe"
) else (
    "%BUILD_DIR%\benchmarks\bench_%ARG%.exe"
)
exit /b %errorlevel%

:cln
rmdir /s /q "%BUILD_DIR%" 2>nul
echo [bolt] cleaned %BUILD_DIR%
exit /b 0
