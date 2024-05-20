import subprocess

epsilon = 0.06641994156659121
n_0_squared = 112.2020615473713

python_script_path = "longRun.py"
python_excecutable_path = r"C:\Users\AADHIL SANU\Desktop\projects\RK2 and Langevin\Scripts\python.exe"


for i in range(2500):
    iter_no = i + 1

    command = [
        python_excecutable_path, python_script_path,
        "--epsilon", str(epsilon),
        "--N_0_squared", str(n_0_squared),
        "--Count", str(iter_no)
    ]

    subprocess.run(command, check=True)