#!/bin/sh

cd "$(dirname "$0")"

isort --profile black ../.. --skip env --skip .venv --skip build --skip deps --skip docker/lib
