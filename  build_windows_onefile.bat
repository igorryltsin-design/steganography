@echo off
setlocal

cd /d "%~dp0"

echo ==========================================
echo   Сборка ONEFILE EXE для Стего Студии
echo ==========================================
echo.

where py >nul 2>nul
if errorlevel 1 (
  echo [ОШИБКА] Python Launcher ^(py^) не найден.
  exit /b 1
)

if not exist ".venv" (
  echo [1/5] Создаю виртуальное окружение...
  py -3 -m venv .venv
  if errorlevel 1 exit /b 1
)

echo [2/5] Активирую виртуальное окружение...
call ".venv\Scripts\activate.bat"
if errorlevel 1 exit /b 1

echo [3/5] Ставлю зависимости...
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
python -m pip install pyinstaller
if errorlevel 1 exit /b 1

echo [4/5] Очищаю старую сборку...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist "СтегоСтудия.spec" del /q "СтегоСтудия.spec"

echo [5/5] Собираю ONEFILE...
pyinstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --windowed ^
  --name "СтегоСтудия" ^
  --collect-all PySide6 ^
  --collect-all pyqtgraph ^
  main_qt.py

if errorlevel 1 (
  echo [ОШИБКА] Сборка завершилась с ошибкой.
  exit /b 1
)

echo.
echo Готово: %CD%\dist\СтегоСтудия.exe
endlocal
exit /b 0
