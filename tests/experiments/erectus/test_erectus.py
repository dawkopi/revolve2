import glob
import os
import subprocess
from tests.conftest import ERECTUS_DIR, TEST_DIR
import uuid


def test_experiment_can_complete():
    """Test that optimize.py can complete (a minimal experiment) without crashing."""
    # assert 1 == 1

    run_name = f"unit_test_{uuid.uuid4()}"
    cmd = [
        "python3",
        os.path.join(ERECTUS_DIR, "optimize.py"),
        "-n",
        run_name,
        "-p",
        "3",
        "--offspring_size",
        "2",
        "-g",
        "1",
        "-t",
        "3",
        # "--save_best",  # don't output best robots
    ]
    print("running command:")
    print(cmd)
    print(" ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE)

    print(res.stdout)
    assert res.returncode == 0
    # database dir gets created relative to directory optimize.py was called from...
    dirs = glob.glob(os.path.join(TEST_DIR, "database/", f"{run_name}*"))
    assert len(dirs) == 1
    exp_dir = dirs[0]
    print("exp_dir = " + exp_dir)

    assert os.path.isfile(os.path.join(exp_dir, "db.sqlite")) == True

    print(res.stdout)
