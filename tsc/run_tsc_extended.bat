@echo off
REM ===================================================================
REM  TSC SF+GPI EXTENDED experiments - Windows workstation
REM
REM  Four additional experiments beyond the baseline run_tsc.bat:
REM    EXP1 hetero : IDDQN retrain-from-scratch on the hetero target task
REM    EXP2 rollouts: multi-rollout variance, 10 rollouts, all methods, no retrain
REM    EXP3 budget : composer episode-budget sweep 100/200/500, homo task
REM    EXP4 ksweep : L1 library-size sweep K=1..5, homo task
REM
REM  Usage:
REM    run_tsc_extended.bat            all four
REM    run_tsc_extended.bat hetero     EXP1 only
REM    run_tsc_extended.bat rollouts   EXP2 only
REM    run_tsc_extended.bat budget     EXP3 only
REM    run_tsc_extended.bat ksweep     EXP4 only
REM
REM  Pre-requisite: run_tsc.bat must have completed for seed 0 first.
REM  Checkpoints sf_s0\ and usfa sf_psi.pth are reused by EXP2/3/4.
REM ===================================================================
setlocal enabledelayedexpansion

set PY=python
set CK0=sf_s0\usfa\sf_psi.pth
set EVAL_EPISODES=5
set EVAL_EPISODES_MULTI=10
set LOGDIR=logs

if not exist %LOGDIR% mkdir %LOGDIR%

set RUN_ALL=1
set RUN_HETERO=0
set RUN_ROLLOUTS=0
set RUN_BUDGET=0
set RUN_KSWEEP=0

if "%1"=="hetero"   (set RUN_ALL=0& set RUN_HETERO=1)
if "%1"=="rollouts" (set RUN_ALL=0& set RUN_ROLLOUTS=1)
if "%1"=="budget"   (set RUN_ALL=0& set RUN_BUDGET=1)
if "%1"=="ksweep"   (set RUN_ALL=0& set RUN_KSWEEP=1)

if %RUN_ALL%==1 (
    set RUN_HETERO=1
    set RUN_ROLLOUTS=1
    set RUN_BUDGET=1
    set RUN_KSWEEP=1
)

REM ================================================================
REM  EXP1 -- Hetero retrain baseline, seed 0
REM ================================================================
if %RUN_HETERO%==1 (
    echo.
    echo ============================================================
    echo  EXP1: Hetero retrain baseline  [seed 0]
    echo ============================================================
    if not exist hetero_retrain mkdir hetero_retrain
    echo [EXP1] Training IDDQN on hetero composite task, seed 0 ...
    %PY% train.py --net gat --reward composite --seed 0 > %LOGDIR%\retrain_hetero.log 2>&1
    if errorlevel 1 goto err
    if exist composite_s0\DQN_gat\pretrain_eval.txt copy /Y composite_s0\DQN_gat\pretrain_eval.txt hetero_retrain\retrain_hetero_eval.txt >nul 2>&1
    echo [EXP1] Done. Check hetero_retrain\retrain_hetero_eval.txt
)

REM ================================================================
REM  EXP2 -- Multi-rollout variance, seed 0, 10 rollouts, no retrain
REM ================================================================
if %RUN_ROLLOUTS%==1 (
    echo.
    echo ============================================================
    echo  EXP2: Multi-rollout variance  [seed 0, 10 rollouts]
    echo ============================================================
    if not exist rollout_variance mkdir rollout_variance
    if not exist "%CK0%" (
        echo ERROR: %CK0% not found. Run run_tsc.bat seed 0 first.
        goto err
    )

    echo [EXP2] L2 transfer homo, 10 rollouts ...
    %PY% train_sf.py --phase transfer --sf_ckpt %CK0% --seed 0 --task_seed 11 --eval_episodes %EVAL_EPISODES_MULTI% > %LOGDIR%\rollout_transfer_homo.log 2>&1
    if errorlevel 1 goto err
    copy /Y sf_s0\transfer\transfer_homo_result.json rollout_variance\transfer_homo_result.json >nul 2>&1

    echo [EXP2] L2 transfer hetero, 10 rollouts ...
    %PY% train_sf.py --phase transfer --sf_ckpt %CK0% --seed 0 --hetero 1 --task_seed 11 --eval_episodes %EVAL_EPISODES_MULTI% > %LOGDIR%\rollout_transfer_hetero.log 2>&1
    if errorlevel 1 goto err
    copy /Y sf_s0\transfer\transfer_hetero_result.json rollout_variance\transfer_hetero_result.json >nul 2>&1

    echo [EXP2] L1 homo, 10 rollouts ...
    %PY% train_sf.py --phase l1 --sf_ckpt %CK0% --seed 0 --task_seed 11 --eval_episodes %EVAL_EPISODES_MULTI% > %LOGDIR%\rollout_l1_homo.log 2>&1
    if errorlevel 1 goto err
    copy /Y sf_s0\l1\l1_homo_result.json rollout_variance\l1_homo_result.json >nul 2>&1

    echo [EXP2] L1 hetero, 10 rollouts ...
    %PY% train_sf.py --phase l1 --sf_ckpt %CK0% --seed 0 --hetero 1 --task_seed 11 --eval_episodes %EVAL_EPISODES_MULTI% > %LOGDIR%\rollout_l1_hetero.log 2>&1
    if errorlevel 1 goto err
    copy /Y sf_s0\l1\l1_hetero_result.json rollout_variance\l1_hetero_result.json >nul 2>&1

    echo [EXP2] L2prime composer homo, 10 rollouts ...
    %PY% train_sf.py --phase composer --sf_ckpt %CK0% --seed 0 --task_seed 11 --episodes 0 --eval_episodes %EVAL_EPISODES_MULTI% > %LOGDIR%\rollout_composer_homo.log 2>&1
    if errorlevel 1 goto err
    copy /Y sf_s0\composer\composer_homo_result.json rollout_variance\composer_homo_result.json >nul 2>&1

    echo [EXP2] L2prime composer hetero, 10 rollouts ...
    %PY% train_sf.py --phase composer --sf_ckpt %CK0% --seed 0 --hetero 1 --task_seed 11 --episodes 0 --eval_episodes %EVAL_EPISODES_MULTI% > %LOGDIR%\rollout_composer_hetero.log 2>&1
    if errorlevel 1 goto err
    copy /Y sf_s0\composer\composer_hetero_result.json rollout_variance\composer_hetero_result.json >nul 2>&1

    echo [EXP2] Done. mean_return / std_return in each JSON under rollout_variance\
)

REM ================================================================
REM  EXP3 -- Composer budget sweep, seed 0, homo task
REM ================================================================
if %RUN_BUDGET%==1 (
    echo.
    echo ============================================================
    echo  EXP3: Composer budget sweep  [seed 0, homo]
    echo ============================================================
    if not exist budget_sweep mkdir budget_sweep
    if not exist "%CK0%" (
        echo ERROR: %CK0% not found. Run run_tsc.bat seed 0 first.
        goto err
    )
    for %%E in (100 200 500) do (
        echo [EXP3] composer homo  episodes=%%E ...
        %PY% train_sf.py --phase composer --sf_ckpt %CK0% --seed 0 --task_seed 11 --episodes %%E --eval_episodes %EVAL_EPISODES% > %LOGDIR%\composer_homo_ep%%E.log 2>&1
        if errorlevel 1 goto err
        copy /Y sf_s0\composer\composer_homo_result.json budget_sweep\composer_homo_ep%%E_result.json >nul 2>&1
        echo [EXP3]   -^> budget_sweep\composer_homo_ep%%E_result.json
    )
    echo [EXP3] Done.
)

REM ================================================================
REM  EXP4 -- L1 library-size K sweep, seed 0, homo task
REM ================================================================
if %RUN_KSWEEP%==1 (
    echo.
    echo ============================================================
    echo  EXP4: L1 library-size K sweep  [seed 0, homo]
    echo ============================================================
    if not exist k_sweep mkdir k_sweep
    if not exist "%CK0%" (
        echo ERROR: %CK0% not found. Run run_tsc.bat seed 0 first.
        goto err
    )

    echo [EXP4] L2 transfer content-independent reference ...
    %PY% train_sf.py --phase transfer --sf_ckpt %CK0% --seed 0 --task_seed 11 --eval_episodes %EVAL_EPISODES% > %LOGDIR%\ksweep_transfer.log 2>&1
    if errorlevel 1 goto err
    copy /Y sf_s0\transfer\transfer_homo_result.json k_sweep\l2_homo_result.json >nul 2>&1

    for %%K in (1 2 3 4 5) do (
        echo [EXP4] L1 homo  lib_size=%%K ...
        %PY% train_sf.py --phase l1 --sf_ckpt %CK0% --seed 0 --task_seed 11 --lib_size %%K --eval_episodes %EVAL_EPISODES% > %LOGDIR%\ksweep_l1_k%%K.log 2>&1
        if errorlevel 1 goto err
        copy /Y sf_s0\l1\l1_homo_result.json k_sweep\l1_homo_k%%K_result.json >nul 2>&1
        echo [EXP4]   -^> k_sweep\l1_homo_k%%K_result.json
    )
    echo [EXP4] Done.
)

echo.
echo ============================================================
echo  All requested experiments complete.
echo    EXP1 hetero retrain   -^> hetero_retrain\
echo    EXP2 rollout variance -^> rollout_variance\
echo    EXP3 budget sweep     -^> budget_sweep\
echo    EXP4 K sweep          -^> k_sweep\
echo ============================================================
goto end

:err
echo.
echo *** A STEP FAILED. Check the matching log in %LOGDIR%\ ***
exit /b 1

:end
endlocal
