package com.hyfly.unit05.entity;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import com.hyfly.utils.Corr2d;

public class ConvolutionalLayer {

    private NDArray w;
    private NDArray b;

    public NDArray getW() {
        return w;
    }

    public NDArray getB() {
        return b;
    }

    public ConvolutionalLayer(Shape shape) {
        try (NDManager manager = NDManager.newBaseManager()) {
            w = manager.create(shape);
            b = manager.randomNormal(new Shape(1));
            w.setRequiresGradient(true);
        }
    }

    public NDArray forward(NDArray X) {
        try (NDManager manager = NDManager.newBaseManager()) {
            return Corr2d.corr2d(manager, X, w).add(b);
        }
    }

}
