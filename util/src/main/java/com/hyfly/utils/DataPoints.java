package com.hyfly.utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DataPoints {

    private static final Logger log = LoggerFactory.getLogger(DataPoints.class);
    private final NDArray X;
    private final NDArray y;

    public DataPoints(NDArray X, NDArray y) {
        this.X = X;
        this.y = y;
    }

    // Generate y = X w + b + noise
    public static DataPoints syntheticData(NDManager manager, NDArray w, float b, int numExamples) {
        NDArray X = manager.randomNormal(new Shape(numExamples, w.size()));
        log.info("x.getShape {}, x.size {}", X.getShape().toString(), X.size());

        NDArray y = X.matMul(w).add(b);
        log.info("y.getShape {}, y.size {}", y.getShape().toString(), y.size());

        // Add noise
        y = y.add(manager.randomNormal(0, 0.01f, y.getShape(), DataType.FLOAT32));
        return new DataPoints(X, y);
    }

    public NDArray getX() {
        return X;
    }

    public NDArray getY() {
        return y;
    }
}
