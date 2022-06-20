# NN-VMD

## How to use

1. ECG5000 Data download from ```timeseriesclassification```

```execute ./data.bat```

2. Install library using ```requirements.txt```

```pip install -r requirements.txt```

3. Execute python file (tensorflow or pytorch)

```python tf_main.py``` or ```python th_main.py```

4. You can modify VMD setting or AI model via 

```./utils/util.py``` and ```./models/model.py```

## Plan

- [ ] Graph neural nets + Shallow neural nets
- [ ] Multi-task learning (e.g. decomposition and classification task)