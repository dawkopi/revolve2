#/bin/bash
# install this repo/dependencies (in dev mode).

set -e

#pip install virtualenv && virtualenv .venv
#. .venv/bin/activate

pip install ./serialization[dev]
pip install ./actor_controller[dev]
pip install ./rpi_controller[dev]
pip install ./core[dev]
pip install ./standard_resources[dev]

#pip install ./runners/isaacgym[dev]
echo "IsaacGym is not open source so sadly we cannot test this."

pip install ./runners/mujoco[dev]
sudo apt install -y libcereal-dev
pip install ./genotypes/cppnwin[dev]

echo "install_dev.sh complete!"