#!/bin/bash

# Call the first script
echo "Running create_env.sh..."
bash create_env.sh

# Check if the first script succeeded
if [ $? -ne 0 ]; then
    echo "create_env.sh failed. Exiting."
    exit 1
fi

# Call the second script after the first completes
echo "Running install_packages.sh..."
bash install_packages.sh

# Check if the second script succeeded
if [ $? -ne 0 ]; then
    echo "install_packages.sh failed. Exiting."
    exit 1
fi

echo "All tasks completed."
