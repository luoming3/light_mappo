#!/bin/bash

set -e

# remote user and host
REMOTE_USER=${1}
REMOTE_HOST=${2}

if [ ! -f ~/.ssh/id_rsa ]; then
    echo "Generating SSH key pair..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -C "" -N ""
fi

echo "Copying public key to remote server..."
ssh-copy-id ${REMOTE_USER}@${REMOTE_HOST}

echo "Testing SSH login without password..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "echo 'hello world'"

echo "SSH no-password setup complete!"
