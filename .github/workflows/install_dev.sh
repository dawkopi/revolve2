#/bin/bash
# install this repo/dependencies (in dev mode).

set -e

#pip install virtualenv && virtualenv .venv
#. .venv/bin/activate

# toggle edit mode (for pip install) on or off
edit=()
if [ $1 == "-e" ]; then
    edit=(-e)
    echo "installing in edit mode!"
else
    echo "installing without edit mode!"
fi

pip install "${edit[@]}" ./serialization[dev]
pip install "${edit[@]}" ./actor_controller[dev]
pip install "${edit[@]}" ./rpi_controller[dev]
pip install "${edit[@]}" ./core[dev]
pip install "${edit[@]}" ./standard_resources[dev]

#pip install ./runners/isaacgym[dev]
echo "IsaacGym is not open source so sadly we cannot test this."

pip install "${edit[@]}" ./runners/mujoco[dev]
sudo apt install -y libcereal-dev
pip install "${edit[@]}" ./genotypes/cppnwin[dev]

echo "install_dev.sh complete!"