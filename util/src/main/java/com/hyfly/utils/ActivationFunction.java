package com.hyfly.utils;

import ai.djl.ndarray.NDList;

public class ActivationFunction {
    public static NDList softmax(NDList arrays) {
        return new NDList(arrays.singletonOrThrow().logSoftmax(1));
    }
}
