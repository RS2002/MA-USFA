@echo off
REM ===================================================================
REM  TSC SF+GPI experiment runner (Windows workstation)
REM  Four methods x two tasks:  retrain / L1 / L2(transfer) / L2'(composer)
REM  on w-homogeneous and w-heterogeneous target tasks.
REM
REM  Usage:   run_tsc.bat                 (uses defaults below)
REM           run_tsc.bat 0 11            (SEED=0, TASK_SEED=11)
REM
REM  Everything is logged to .\logs\ ; per-method results land in
REM  .\sf_s%SEED%\<phase>\<phase>_<homo|hetero>_result.json
REM ===================================================================
setlocal enabledelayedexpansion

REM ---- config (override via args) ----
set SEED=%1
if "%SEED%"=="" set SEED=0
set TASK_SEED=%2
if "%TASK_SEED%"=="" set TASK_SEED=11

set USFA_EPISODES=500
set COMPOSER_EPISODES=200
set EVAL_EPISODES=5

set PY=python
set CK=sf_s%SEED%\usfa\sf_psi.pth
set LOGDIR=logs
if not exist %LOGDIR% mkdir %LOGDIR%

echo ============================================================
echo  SEED=%SEED%  TASK_SEED=%TASK_SEED%
echo  usfa_ep=%USFA_EPISODES%  composer_ep=%COMPOSER_EPISODES%  eval=%EVAL_EPISODES%
echo  checkpoint = %CK%
echo ============================================================

REM ---- 0. w-free successor features (shared by L1/L2/L2') ----
if exist "%CK%" (
    echo [0/8] usfa checkpoint already exists, skipping training: %CK%
) else (
    echo [0/8] training w-free successor features ...
    %PY% train_sf.py --phase usfa --seed %SEED% --episodes %USFA_EPISODES% > %LOGDIR%\usfa.log 2>&1
    if errorlevel 1 goto :err
)
if not exist "%CK%" (
    echo ERROR: expected checkpoint not found after usfa: %CK%
    goto :err
)

REM ================= w-HOMOGENEOUS target task =================
echo [1/8] L2 transfer  (homogeneous) ...
%PY% train_sf.py --phase transfer --sf_ckpt "%CK%" --seed %SEED% --task_seed %TASK_SEED% --eval_episodes %EVAL_EPISODES% > %LOGDIR%\transfer_homo.log 2>&1
if errorlevel 1 goto :err

echo [2/8] L1 anchor    (homogeneous) ...
%PY% train_sf.py --phase l1 --sf_ckpt "%CK%" --seed %SEED% --task_seed %TASK_SEED% --eval_episodes %EVAL_EPISODES% > %LOGDIR%\l1_homo.log 2>&1
if errorlevel 1 goto :err

echo [3/8] L2' composer (homogeneous) ...
%PY% train_sf.py --phase composer --sf_ckpt "%CK%" --seed %SEED% --task_seed %TASK_SEED% --episodes %COMPOSER_EPISODES% --eval_episodes %EVAL_EPISODES% > %LOGDIR%\composer_homo.log 2>&1
if errorlevel 1 goto :err

echo [4/8] retrain IDDQN (homogeneous baseline) ...
%PY% train.py --net gat --reward composite --seed %SEED% > %LOGDIR%\retrain.log 2>&1
if errorlevel 1 goto :err

REM ================= w-HETEROGENEOUS target task =================
echo [5/8] L2 transfer  (heterogeneous) ...
%PY% train_sf.py --phase transfer --sf_ckpt "%CK%" --seed %SEED% --hetero 1 --task_seed %TASK_SEED% --eval_episodes %EVAL_EPISODES% > %LOGDIR%\transfer_hetero.log 2>&1
if errorlevel 1 goto :err

echo [6/8] L1 anchor    (heterogeneous) ...
%PY% train_sf.py --phase l1 --sf_ckpt "%CK%" --seed %SEED% --hetero 1 --task_seed %TASK_SEED% --eval_episodes %EVAL_EPISODES% > %LOGDIR%\l1_hetero.log 2>&1
if errorlevel 1 goto :err

echo [7/8] L2' composer (heterogeneous) ...
%PY% train_sf.py --phase composer --sf_ckpt "%CK%" --seed %SEED% --hetero 1 --task_seed %TASK_SEED% --episodes %COMPOSER_EPISODES% --eval_episodes %EVAL_EPISODES% > %LOGDIR%\composer_hetero.log 2>&1
if errorlevel 1 goto :err

REM ---- 8. collect every result json into one folder ----
echo [8/8] collecting results ...
set OUT=results_s%SEED%
if not exist %OUT% mkdir %OUT%
copy /Y sf_s%SEED%\transfer\transfer_homo_result.json   %OUT%\ >nul 2>&1
copy /Y sf_s%SEED%\l1\l1_homo_result.json               %OUT%\ >nul 2>&1
copy /Y sf_s%SEED%\composer\composer_homo_result.json   %OUT%\ >nul 2>&1
copy /Y sf_s%SEED%\transfer\transfer_hetero_result.json %OUT%\ >nul 2>&1
copy /Y sf_s%SEED%\l1\l1_hetero_result.json             %OUT%\ >nul 2>&1
copy /Y sf_s%SEED%\composer\composer_hetero_result.json %OUT%\ >nul 2>&1
xcopy /Y /I %LOGDIR% %OUT%\logs >nul 2>&1

echo.
echo ============================================================
echo  DONE. Results copied to: %OUT%\
echo  Zip that folder (plus the retrain output dir) and send it back.
echo ============================================================
goto :eof

:err
echo.
echo *** A STEP FAILED (errorlevel %errorlevel%). Check the matching log in %LOGDIR%\ ***
echo *** Fix the error and re-run; the usfa checkpoint is reused automatically. ***
exit /b 1
