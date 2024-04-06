package com.hyfly.utils;

import ai.djl.ndarray.NDArray;

public class Softmax {

    public static NDArray softmax(NDArray X) {
        NDArray Xexp = X.exp();
        NDArray partition = Xexp.sum(new int[]{1}, true);
        return Xexp.div(partition); // 这里应用了广播机制
    }
}
