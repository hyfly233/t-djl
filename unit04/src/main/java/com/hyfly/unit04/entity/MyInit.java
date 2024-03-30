package com.hyfly.unit04.entity;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.initializer.Initializer;

public class MyInit implements Initializer {

    public MyInit() {
    }

    @Override
    public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
        System.out.printf("Init %s\n", shape.toString());
        // Here we generate data points
        // from a uniform distribution [-10, 10]
        NDArray data = manager.randomUniform(-10, 10, shape, dataType);
        // We keep the data points whose absolute value is >= 5
        // and set the others to 0.
        // This generates the distribution `w` shown above.
        NDArray absGte5 = data.abs().gte(5); // returns boolean NDArray where
        // true indicates abs >= 5 and
        // false otherwise
        return data.mul(absGte5); // keeps true indices and sets false indices to 0.
        // special operation when multiplying a numerical
        // NDArray with a boolean NDArray
    }

}
