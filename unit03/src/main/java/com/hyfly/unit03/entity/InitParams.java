package com.hyfly.unit03.entity;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

public class InitParams {

    private NDArray w;
    private NDArray b;
    private NDList l;

    public InitParams() {
        NDManager manager = NDManager.newBaseManager();
//        w = manager.randomNormal(0, 1.0f, new Shape(numInputs, 1), DataType.FLOAT32);
        b = manager.zeros(new Shape(1));
        w.setRequiresGradient(true);
        b.setRequiresGradient(true);
    }

    public NDArray getW() {
        return this.w;
    }

    public NDArray getB() {
        return this.b;
    }
}
