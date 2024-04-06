package com.hyfly.utils;

import ai.djl.ndarray.NDList;

public class Updater {
    public static void updater(NDList params, float lr, int batchSize) {
        Training.sgd(params, lr, batchSize);
    }
}
