#!/usr/bin/env bash
echo "begin to download models"
bash download.sh

echo "begin to unpack models"
mkdir models
tar xvf models.tar -C models

echo "files in current dir"
for entry in "$PWD"/*
do
  echo "$entry"
done

echo "begin to test"
python3 inference.py --local 1