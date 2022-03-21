# Installation
- Install docker (https://www.docker.com/)
- Build the image
In the directory where the Dockerfile is, run:
```
    docker build -t vapo_image .
```

- Run a container
The following line will run the vapo_image under the docker container named "vapo_container". This will open a command line of an ubuntu filesystem in which you can run the code of this repo. The "--rm" flag removes the container after the session is closed.
```
    docker run -it --rm --name vapo_container bash vapo_image
```

# Docker commands
- See current images in system
'''
    docker images
'''

- See running containers
'''
    docker ps
'''