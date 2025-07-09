#!/bin/bash

set -e

echo "Starting test script..."

echo "running test_1s2.sh"
bash scripts/test_1s2.sh
echo "completed test_1s2.sh"

echo "running test_1s3.sh"
bash scripts/test_1s3.sh
echo "completed test_1s3.sh"

echo "running test_1s5.sh"
bash scripts/test_1s5.sh
echo "completed test_1s5.sh"

echo "running test_5s2.sh"
bash scripts/test_5s2.sh
echo "completed test_5s2.sh"

echo "running test_5s3.sh"
bash scripts/test_5s3.sh
echo "completed test_5s3.sh"

echo "running test_5s5.sh"
bash scripts/test_5s5.sh
echo "completed test_5s5.sh"


