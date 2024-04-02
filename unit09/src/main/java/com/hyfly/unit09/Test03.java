package com.hyfly.unit09;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import com.hyfly.utils.attention.MultiHeadAttention;

public class Test03 {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            int numHiddens = 100;
            int numHeads = 5;
            MultiHeadAttention attention = new MultiHeadAttention(numHiddens, numHeads, 0.5f, false);

            //
            int batchSize = 2;
            int numQueries = 4;
            int numKvpairs = 6;
            NDArray validLens = manager.create(new float[]{3, 2});
            NDArray X = manager.ones(new Shape(batchSize, numQueries, numHiddens));
            NDArray Y = manager.ones(new Shape(batchSize, numKvpairs, numHiddens));

            ParameterStore ps = new ParameterStore(manager, false);
            NDList input = new NDList(X, Y, Y, validLens);
            attention.initialize(manager, DataType.FLOAT32, input.getShapes());
            NDList result = attention.forward(ps, input, false);
            result.get(0).getShape();
        }
    }

    public static NDArray transposeQkv(NDArray X, int numHeads) {
        // Shape of input `X`:
        // (`batchSize`, no. of queries or key-value pairs, `numHiddens`).
        // Shape of output `X`:
        // (`batchSize`, no. of queries or key-value pairs, `numHeads`,
        // `numHiddens` / `numHeads`)
        X = X.reshape(X.getShape().get(0), X.getShape().get(1), numHeads, -1);

        // Shape of output `X`:
        // (`batchSize`, `numHeads`, no. of queries or key-value pairs,
        // `numHiddens` / `numHeads`)
        X = X.transpose(0, 2, 1, 3);

        // Shape of `output`:
        // (`batchSize` * `numHeads`, no. of queries or key-value pairs,
        // `numHiddens` / `numHeads`)
        return X.reshape(-1, X.getShape().get(2), X.getShape().get(3));
    }

    public static NDArray transposeOutput(NDArray X, int numHeads) {
        X = X.reshape(-1, numHeads, X.getShape().get(1), X.getShape().get(2));
        X = X.transpose(0, 2, 1, 3);
        return X.reshape(X.getShape().get(0), X.getShape().get(1), -1);
    }
}
