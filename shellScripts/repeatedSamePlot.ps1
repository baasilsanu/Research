$VENV_PATH = "C:\Users\AADHIL SANU\Desktop\projects\RK2 and Langevin\Scripts\Activate.ps1"


$epsilon = .015

$N_0_squared = 100

& $VENV_PATH


for ($i = 0; $i -lt 100; $i++) {
    $iterNo = $i + 1
    Write-Host "Running simulation iteration = $iterNo"
    python repeatedPlotter.py --epsilon $epsilon --N_0_squared $N_0_squared --Count $iterNo
}


