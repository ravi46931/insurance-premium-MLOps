# PowerShell script to set up a Python virtual environment

# Function to check if a command exists
function CommandExists {
    param (
        [string]$command
    )
    $commandPath = (Get-Command $command -ErrorAction SilentlyContinue).Path
    return $commandPath -ne $null
}

# Function to read JSON file
function Read-Json {
    param (
        [string]$jsonFile
    )
    $jsonContent = Get-Content $jsonFile -Raw | ConvertFrom-Json
    return $jsonContent
}

# Load configuration from config.json
$config = Read-Json "config.json"


$template = $config.template
$envDir = $config.envDir
$requirementsFile = $config.requirementsFile

# Check if Python is installed
if (-not (CommandExists "python")) {
    Write-Host "Python is not installed. Please install Python and try again."
    exit 1
}

# Creating Project structure
Write-Host "Creating project structure"
python $template

# Create the virtual environment if it doesn't exist
if (-not (Test-Path $envDir)) {
    Write-Host "Creating virtual environment..."
    python -m venv $envDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtual environment."
        exit 1
    }
} else {
    Write-Host "Virtual environment already exists."
}

# Activate the virtual environment
Write-Host "Activating virtual environment..."
& "$envDir\Scripts\Activate.ps1"

# Check if requirements.txt exists and install packages
if (Test-Path $requirementsFile) {
    Write-Host "Installing packages from requirements.txt..."
    pip install -r $requirementsFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install required packages."
        exit 1
    }
} else {
    Write-Host "No requirements.txt file found. Skipping package installation."
}

Write-Host "Setup complete. Virtual environment is ready."
