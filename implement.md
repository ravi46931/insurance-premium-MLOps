dvc
# Task: 

 1. Complete the process of docker-compose.yaml file (Add both) 
 2. Write the mkdocs and ci.yaml for the mkdocs [DONE]
 3. Write the README file 
 4. Write the process of implementing the project (for self reference)
 5. Write the commands
 6. Arrage the link of all (e.g. github, dagshub etc)
 7. Create a docker image upload to the AWS and make the web app there [DONE]
 8. Or use github action to do all this kind o the things [DONE]

 # Deploy

 ## Create IAM user add the security key and id to gitgub secret keys
 ## Now create an instance and connect it with github self hosted runner (commands are below)
 ## Commands for EC2 terminal
 aws EC2 instance terminal commands

sudo apt-get update -y
sudo apt-get upgrade

curl -fsSL https://get.docker.com -o get-docker.sh

<!-- install docker -->
sudo sh get-docker.sh 
sudo usermod -aG docker ubuntu
newgrp docker

docker images

<!-- Verify contrainer is running -->
docker ps -a

<!-- copy the command from the github/actions/ runners -->

mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.314.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.314.1/actions-runner-linux-x64-2.314.1.tar.gz

tar xzf ./actions-runner-linux-x64-2.314.1.tar.gz
./config.sh --url https://github.com/ravi46931/Xray-image-classification --token A2TWNCAMCQKPIVDZAGJKYK3GCASNE

./run.sh

