package com.hyfly.unit07;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class Test04 {

    public static void main(String[] args) throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray X = manager.randomNormal(0, 1, new Shape(3, 1), DataType.FLOAT32);
            NDArray W_xh = manager.randomNormal(0, 1, new Shape(1, 4), DataType.FLOAT32);
            NDArray H = manager.randomNormal(0, 1, new Shape(3, 4), DataType.FLOAT32);
            NDArray W_hh = manager.randomNormal(0, 1, new Shape(4, 4), DataType.FLOAT32);
            X.dot(W_xh).add(H.dot(W_hh));

            //
            X.concat(H, 1).dot(W_xh.concat(W_hh, 0));
        }
    }
}
