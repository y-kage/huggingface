# 1. Preparation
## Install
### Docker
Follow the instructions
- [Docker Engine](https://docs.docker.com/engine/install/ubuntu/)
- [docker group](https://docs.docker.com/engine/install/linux-postinstall/)

### NVIDIA Container Toolkit
Follow the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Docker Image
- [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda/tags?page=1&ordering=-name)

# 2. Getting Started
## Before Build
Modify files. Be careful os version and python version. \
(ex. Ubuntu 20.04 -> python 3.9, Ubuntu 22.04 -> python 3.10) \
See [here](https://vegastack.com/tutorials/how-to-install-python-3-9-on-ubuntu-22-04/) or [here](https://qiita.com/murakami77/items/b612734ff209cbb22afb)

- Modify `docker_template.env`
  - COMPOSE_PROJECT_NAME : project name
  - UID : UID
  - GID : GID
  - USER_NAME : user name used in container
  - WORKDIR_CONTAINER : container WORKDIR, directory where local WORKDIR mounted to
  - WORKDIR_LOCAL : local WORKDIR, directory mounted to container
  - BASE_IMAGE : Docker Image
  - PYTHON_VERSION : Python version
  - IMAGE_LABEL : label to cache docker image with label, if exist, load, if not, build.
  - CONTAINER_NAME : CONTAINER NAME, used to get in the container
  - HOST_PORT : HOST_PORT, use port not used at other containers
  - CONTAINER_PORT : CONTAINER_PORT, use port not used at other containers
  
  Commands to search your UID, GID
  ```bash
  id -u # UID
  id -g # GID
  ```


- docker-compose.yaml
  Change image, container_name, volumes, shm_size if needed.
  - image : name of image cached to local. if exist, load, if not, build.
  - container_name : name used to get in the container.
  - volumes : Correspondence. {local_dir}:{container_dir}
  - shm_size : shared memory size. check your spec.


- Dockerfile
  Change apt libraries, Pytorch.


## Useful Commands

- Docker compose up \
  Use command at the directory where docker-compose.yaml is
  ```bash
  docker compose up -d
  ```
  
  If Dockerfile changed, Docker compose up with build
  ```bash
  docker compose up -d --build
  ```

- Execute command in Docker
  ```bash
  docker exec -it {container_name} bash
  # or
  docker exec -it -w {WORK_DIR_PATH} {container_name} bash
  # example
  docker exec -it template bash
  ```

  As root
  ```bash
  docker exec -it -u 0 -w {WORK_DIR_PATH} {container_name} bash
  ```

- Using JupyterLab (Optional)
  ```bash
  python -m jupyterlab --ip 0.0.0.0 --port {CONTAINER_PORT} --allow-root
  ```

- Using Tensorboard
  ```bash
  tensorboard --logdir=/workspace/PytorchLightning/lightning_logs --host=0.0.0.0 --port={CONTAINER_PORT}
  # or
  python /home/{USER}/.local/lib/python3.9/site-packages/tensorboard/main.py --logdir=/workspace/PytorchLightning/lightning_logs --host=0.0.0.0 --port={CONTAINER_PORT}
  ```

- Login W & D
  ```bash
  wandb login
  # or
  python3 -m wandb login
  # or
  /usr/bin/python3 -m wandb login
  ```

# 3. Reference
- [Ueda's Sample](https://github.com/sh1027/docker_pytorch)
- [Docker to Ubuntu](https://zenn.dev/usagi1975/articles/2022-09-05-000000_docker_gpu)
- [About Rootless mode](https://qiita.com/boocsan/items/781ae06fa4ac4291ba97)
