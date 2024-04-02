package com.hyfly.unit09;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import com.hyfly.unit09.entity.AdditiveAttention;
import com.hyfly.utils.PlotUtils;
import com.hyfly.utils.attention.DotProductAttention;

public class Test02 {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            maskedSoftmax(
                    manager.randomUniform(0, 1, new Shape(2, 2, 4)),
                    manager.create(new float[]{2, 3}));

            maskedSoftmax(
                    manager.randomUniform(0, 1, new Shape(2, 2, 4)),
                    manager.create(new float[][]{{1, 3}, {2, 4}}));

            NDArray queries = manager.randomNormal(0, 1, new Shape(2, 1, 20), DataType.FLOAT32);
            NDArray keys = manager.ones(new Shape(2, 10, 2));
            // The two value matrices in the `values` minibatch are identical
            NDArray values = manager.arange(40f).reshape(1, 10, 4).repeat(0, 2);
            NDArray validLens = manager.create(new float[]{2, 6});

            AdditiveAttention attention = new AdditiveAttention(8, 0.1f);
            NDList input = new NDList(queries, keys, values, validLens);
            ParameterStore ps = new ParameterStore(manager, false);
            attention.initialize(manager, DataType.FLOAT32, input.getShapes());
            attention.forward(ps, input, false).head();

            //
            PlotUtils.showHeatmaps(
                    attention.attentionWeights.reshape(1, 1, 2, 10),
                    "Keys",
                    "Queries",
                    new String[]{""},
                    500,
                    700);

            //
            queries = manager.randomNormal(0, 1, new Shape(2, 1, 2), DataType.FLOAT32);
            DotProductAttention productAttention = new DotProductAttention(0.5f);
            input = new NDList(queries, keys, values, validLens);
            productAttention.initialize(manager, DataType.FLOAT32, input.getShapes());
            productAttention.forward(ps, input, false).head();

            PlotUtils.showHeatmaps(
                    productAttention.attentionWeights.reshape(1, 1, 2, 10),
                    "Keys",
                    "Queries",
                    new String[] {""},
                    500,
                    700);
        }
    }

    public static NDArray maskedSoftmax(NDArray X, NDArray validLens) {
        /* Perform softmax operation by masking elements on the last axis. */
        // `X`: 3D NDArray, `validLens`: 1D or 2D NDArray
        if (validLens == null) {
            return X.softmax(-1);
        }

        Shape shape = X.getShape();
        if (validLens.getShape().dimension() == 1) {
            validLens = validLens.repeat(shape.get(1));
        } else {
            validLens = validLens.reshape(-1);
        }
        // On the last axis, replace masked elements with a very large negative
        // value, whose exponentiation outputs 0
        X = X.reshape(new Shape(-1, shape.get(shape.dimension() - 1)))
                .sequenceMask(validLens, (float) -1E6);
        return X.softmax(-1).reshape(shape);
    }
}
