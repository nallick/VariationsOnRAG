#!/bin/bash

mkdir -p deploy
cp -f *.py deploy
export TMPDIR=$(pwd)/pip_tmp # force pip to use our directory instead of /tmp because EC2 makes /tmp too small
# pip install torch --extra-index-url https://download.pytorch.org/whl/cpu --python-version 3.12 --only-binary=:all: --platform manylinux2014_x86_64 --target ./deploy --cache-dir=pip_cache
# pip install -r requirements.txt --python-version 3.12 --only-binary=:all: --platform manylinux2014_x86_64 --target ./deploy --cache-dir=pip_cache
# pip install -r requirements.txt --python-version 3.12 --only-binary=:all: --platform manylinux2014_aarch64 --target ./deploy --cache-dir=pip_cache
cd deploy
zip -q -r -9 ../deploy.zip *
cd ..
# rm -f -R deploy
# aws s3 cp deploy.zip s3://postgres-retrieval-service-deployment/deploy.zip
