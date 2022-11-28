import glob
import os
import subprocess

from tests.conftest import DATABASE_PATH, ERECTUS_DIR, TEST_DIR, get_uuid


EXP_CMD_BASE = [
    "python3",
    os.path.join(ERECTUS_DIR, "optimize.py"),
    "-p",
    "16",
    "--offspring_size",
    "16",
    "-g",
    "1",
    "-t",
    "3",
    # "--save_best",  # don't output best robots
]


def test_experiment_can_complete():
    """Test that optimize.py can complete (a minimal experiment) without crashing."""
    # assert 1 == 1

    run_name = f"unit_test_{get_uuid()}"
    cmd = EXP_CMD_BASE.copy() + [
        "-n",
        run_name,
    ]
    print("running command:")
    print(cmd)
    print(" ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE)

    print(res.stdout)
    assert res.returncode == 0
    # database dir gets created relative to directory optimize.py was called from...
    pattern = os.path.join(DATABASE_PATH, f"{run_name}*")
    dirs = glob.glob(pattern)
    assert len(dirs) == 1
    exp_dir = dirs[0]
    print("exp_dir = " + exp_dir)

    assert os.path.isfile(os.path.join(exp_dir, "db.sqlite")) == True

    print(res.stdout)


def test_experiment_can_complete__cma_es():
    """Test that optimize.py can complete (a minimal experiment) without crashing."""
    # assert 1 == 1

    run_name = f"unit_test_{get_uuid()}"
    cmd = EXP_CMD_BASE.copy() + [
        "-n",
        run_name,
        "--use_cma",
    ]
    print("running command:")
    print(cmd)
    print(" ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE)

    print(res.stdout)
    assert res.returncode == 0
    # database dir gets created relative to directory optimize.py was called from...
    pattern = os.path.join(DATABASE_PATH, f"{run_name}*")
    dirs = glob.glob(pattern)
    assert len(dirs) == 1
    exp_dir = dirs[0]
    print("exp_dir = " + exp_dir)

    assert os.path.isfile(os.path.join(exp_dir, "db.sqlite")) == True

    print(res.stdout)
