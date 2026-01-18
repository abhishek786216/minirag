@echo off
echo 🚀 Mini RAG Application Setup
echo ==============================

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo ✅ Python found: 
python --version

:: Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not installed. Please install pip and try again.
    pause
    exit /b 1
)

echo ✅ pip found:
pip --version

:: Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

:: Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo 📥 Installing Python dependencies...
pip install -r requirements.txt

:: Create .env file if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo ⚠️  Please edit .env file with your API keys:
    echo    - OPENAI_API_KEY
    echo    - PINECONE_API_KEY
    echo    - PINECONE_ENVIRONMENT
    echo    - COHERE_API_KEY
    echo.
)

:: Check if .env file has been configured
findstr "your_.*_key_here" .env >nul
if not errorlevel 1 (
    echo ⚠️  Warning: .env file still contains placeholder values.
    echo    Please update your API keys before running the application.
    echo.
)

echo ✅ Setup complete!
echo.
echo Next steps:
echo 1. Edit .env file with your API keys
echo 2. Run: venv\Scripts\activate.bat
echo 3. Run: python main.py
echo 4. Open: http://localhost:8000
echo.
echo Happy coding! 🎉
pause