$VENV_PATH = "C:\Users\AADHIL SANU\Desktop\projects\RK2 and Langevin\Scripts\Activate.ps1"


$epsilons = @(.177458622221407, .223740466599542, .01)

$N_0_squareds = @(581.551898100832, 857.239499589027, 100)

& $VENV_PATH


for ($i = 0; $i -lt $epsilons.Length; $i++) {
    $epsilon = $epsilons[$i]
    $N_0_squared = $N_0_squareds[$i]
    Write-Host "Running simulation with epsilon=$epsilon and N_0_squared=$N_0_squared"
    python ProjectShellWork.py --epsilon $epsilon --N_0_squared $N_0_squared
}


