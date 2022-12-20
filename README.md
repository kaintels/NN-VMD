# NN-VMD

## Prerequisite

- Python 3.7+
- Julia 1.7+

## How to use

1. ECG5000 Data download from ```timeseriesclassification``` Execute ```./data.bat```

2. Install library using ```pip install -r requirements.txt```

3. Execute ```julia requirement.jl``` (install lib)

4. set python path for PyCall.jl
```
ENV["PYTHON"] = raw"C:your_python_env/python.exe"
Pkg.build("PyCall")
```

5. Execute python file ```python main.py```

4. You can modify VMD setting or AI model via 

```./utils/util.py``` and ```./utils/preprocessing.jl``` and ```./models/model.py```

## Plan
- [x] VAE (Variational Auto Encoder)
- [ ] Graph neural nets + Shallow neural nets
- [x] Multi-task learning (e.g. decomposition and classification task)