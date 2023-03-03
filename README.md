# NN-VMD

## Prerequisite

- Python 3.7+
- Julia 1.7+
- IF use docker gpu, you should install ```nvidia-cuda-toolkit``` and ```nvidia-container-toolkit```

```
sudo apt install -y nvidia-cuda-toolkit # nvidia-cuda-toolkit installation
```

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit # nvidia-container-toolkit installation

sudo /etc/init.d/docker restart
```


## How to use (Windows)
<details>
<div markdown="1">

1. ECG5000 Data download from ```timeseriesclassification``` Execute ```./data.bat```

2. Install library using ```pip install -r requirements.txt```

3. Execute ```julia requirements.jl``` (install lib)

4. Execute ```python julia_setting.py```

5. Execute python file ```python main.py```

4. You can modify VMD setting or AI model via 

```./utils/util.py``` and ```./utils/preprocessing.jl``` and ```./models/model.py```

</div>
</details>

## How to use (Linux)
<details>
<div markdown="1">

1. ECG5000 Data download from ```timeseriesclassification``` Execute ```sh data.sh```

2. Install library using ```pip install -r requirements.txt```

3. Execute ```julia requirements.jl``` (install lib)

4. Execute ```python julia_setting.py```

5. Execute python file ```python main.py```

4. You can modify VMD setting or AI model via 

```./utils/util.py``` and ```./utils/preprocessing.jl``` and ```./models/model.py```

</div>
</details>

## How to use (Docker)
<details>
<div markdown="1">

1. if Docker turn off, Execute ```sudo service docker start```

2. Execute ```docker build -t nn-vmd .```

3. Execute **GPU version**
   ```docker run -it --gpus all --name nn-vmd nn-vmd:latest bash train.sh```(default : MTL)

   Execute **CPU version**
   ```docker run -it --name nn-vmd nn-vmd:latest bash train.sh```

4. Option Execute

```
docker start nn-vmd (required)
docker exec -it nn-vmd bash train.sh cnn
docker exec -it nn-vmd bash train.sh vae
```

</div>
</details>


## Plan
- [x] VAE (Variational Auto Encoder)
- [ ] Graph neural nets + Shallow neural nets
- [x] Multi-task learning (e.g. decomposition and classification task)