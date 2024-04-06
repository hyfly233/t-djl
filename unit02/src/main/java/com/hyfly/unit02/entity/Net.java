package com.hyfly.unit02.entity;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import com.hyfly.utils.Softmax;

// We need to wrap `net()` in a class so that we can reference the method
// and pass it as a parameter to a function or save it in a variable
public class Net {


    // 3.6.3. 定义模型
    public static NDArray net(NDList params, NDArray X, int numInputs) {
        NDArray currentW = params.get(0);
        NDArray currentB = params.get(1);
        return Softmax.softmax(X.reshape(new Shape(-1, numInputs)).dot(currentW).add(currentB));
    }
}
