# NN-VMD

## Prerequisite

- Python 3.7+
- Julia 1.7+

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

3. Execute ```docker run -it --gpus all nn-vmd:latest bash train.sh```(default : MTL)

4. Option Execute

```
docker start (required)
docker exec -it nn-vmd bash train.sh cnn
docker exec -it nn-vmd bash train.sh vae
```

</div>
</details>


## Plan
- [x] VAE (Variational Auto Encoder)
- [ ] Graph neural nets + Shallow neural nets
- [x] Multi-task learning (e.g. decomposition and classification task)