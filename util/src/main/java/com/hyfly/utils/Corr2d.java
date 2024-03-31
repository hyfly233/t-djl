package com.hyfly.utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;

public class Corr2d {

    public static NDArray corr2d(NDManager manager, NDArray X, NDArray K) {
        // 计算二维互关联。
        int h = (int) K.getShape().get(0);
        int w = (int) K.getShape().get(1);

        NDArray Y = manager.zeros(new Shape(X.getShape().get(0) - h + 1, X.getShape().get(1) - w + 1));

        for (int i = 0; i < Y.getShape().get(0); i++) {
            for (int j = 0; j < Y.getShape().get(1); j++) {
                Y.set(new NDIndex(i + "," + j), X.get(i + ":" + (i + h) + "," + j + ":" + (j + w)).mul(K).sum());
            }
        }

        return Y;
    }
}
