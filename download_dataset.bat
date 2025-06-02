@echo off
setlocal

:: === Ayarlar ===
set "url=https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip"
set "output=MSRVTT.zip"
set "extract_dir=MSRVTT"

:: === İndirme İşlemi ===

echo [INFO] MSRVTT dataset is downloading...

:: curl varsa kullan
where curl >nul 2>&1
if %errorlevel%==0 (
    echo [INFO] curl found, installation starting...
    curl -L -o "%output%" "%url%"
    goto :extract
)

:: bitsadmin varsa kullan
where bitsadmin >nul 2>&1
if %errorlevel%==0 (
    echo [INFO] curl not found, installation starting with bitsadmin...
    bitsadmin /transfer myDownloadJob /download /priority normal "%url%" "%cd%\%output%"
    goto :extract
)

:: ikisi de yoksa hata ver
echo [ERROR] you need curl or bitsadmin for downloading the dataset.
echo for install curl: https://curl.se/windows/
goto :end

:: === Çıkarma İşlemi ===
:extract
echo [INFO] downloading done. Extracting zip file...

:: Çıkarma klasörü varsa silinmesin
if not exist "%extract_dir%" mkdir "%extract_dir%"

:: tar komutu varsa kullan
where tar >nul 2>&1
if %errorlevel%==0 (
    tar -xf "%output%" -C "%extract_dir%"
    echo [INFO] Extracting done: %extract_dir% folder.
    goto :done
)

:: tar yoksa powershell ile çıkar
powershell -Command "Expand-Archive -LiteralPath '%output%' -DestinationPath '%extract_dir%' -Force"
echo [INFO] Extracting done (Used PowerShell): %extract_dir% dir.
goto :done

:done
echo [INFO] Operation done. ZIP File: %output%, Directory: %extract_dir%
pause

:end
endlocal
