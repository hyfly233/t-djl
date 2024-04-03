package com.hyfly.unit07.entity;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.util.Pair;
import com.hyfly.utils.Functions;

/**
 * 从头开始实现的RNN模型
 */
public class RNNModelScratch {
    public int vocabSize;
    public int numHiddens;
    public NDList params;
    public Functions.TriFunction<Integer, Integer, Device, NDList> initState;
    public Functions.TriFunction<NDArray, NDList, NDList, Pair> forwardFn;

    public RNNModelScratch(
            int vocabSize,
            int numHiddens,
            Device device,
            Functions.TriFunction<Integer, Integer, Device, NDList> getParams,
            Functions.TriFunction<Integer, Integer, Device, NDList> initRNNState,
            Functions.TriFunction<NDArray, NDList, NDList, Pair> forwardFn) {
        this.vocabSize = vocabSize;
        this.numHiddens = numHiddens;
        this.params = getParams.apply(vocabSize, numHiddens, device);
        this.initState = initRNNState;
        this.forwardFn = forwardFn;
    }

    public Pair forward(NDArray X, NDList state) {
        X = X.transpose().oneHot(this.vocabSize);
        return this.forwardFn.apply(X, state, this.params);
    }

    public NDList beginState(int batchSize, Device device) {
        return this.initState.apply(batchSize, this.numHiddens, device);
    }
}
