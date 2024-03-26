package com.hyfly.unit02.entity;

import ai.djl.ndarray.NDArray;

public class DataPoints {
    private NDArray X, y;

    public DataPoints(NDArray X, NDArray y) {
        this.X = X;
        this.y = y;
    }

    public NDArray getX() {
        return X;
    }

    public NDArray getY() {
        return y;
    }
}
