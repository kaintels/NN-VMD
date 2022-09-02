using PyCall
using VMD
io = pyimport("scipy.io")

train_path = "../dataset/ECG5000_TRAIN.arff"
test_path = "../dataset/ECG5000_TEST.arff"

py"""
import numpy as np
from scipy.io import arff
import pandas as pd

def data_load(path):
    data = arff.loadarff(path)
    output = pd.DataFrame(data[0], dtype=np.float32)

    return output.values
"""

function vmd_calculate(path, output_name)
    data = py"""data_load"""(path)
    alpha = 2000;       # moderate bandwidth constraint
    tau = 0;            # noise-tolerance (no strict fidelity enforcement)
    K = 3;              # 3 modes
    tol = 1e-7;
    sample_frequency = 140;

    output_ch1 = []
    output_ch2 = []
    output_ch3 = []

    output = []
    for i=1:1:length(data[:,1])

        v = vmd(data[i, 1:140] ; 
        alpha = alpha,
        tau = tau,
        K = K,
        DC = false,
        init = 1,
        tol = tol, 
        sample_frequency = sample_frequency);

        output_ch1 = push!(output_ch1, v.signal_d[:, 1]);
        output_ch2 = push!(output_ch2, v.signal_d[:, 2]);
        output_ch3 = push!(output_ch3, v.signal_d[:, 3]);
    end

    output_ch1 = reduce(vcat, transpose.(output_ch1));
    output_ch2 = reduce(vcat, transpose.(output_ch2));
    output_ch3 = reduce(vcat, transpose.(output_ch3));
    output = cat(output_ch1, output_ch2, output_ch3, dims=3);


    io.savemat("../dataset/$output_name.mat", Dict("data" => output));
end


cd("utils")

vmd_calculate(train_path, "processed_train")
vmd_calculate(test_path, "processed_test")

cd("../")