package com.hyfly.unit09.entity;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.hyfly.unit09.Test02;

/* Additive attention. */
public class AdditiveAttention extends AbstractBlock {

    private Linear W_k;
    private Linear W_q;
    private Linear W_v;
    private Dropout dropout;
    public NDArray attentionWeights;

    public AdditiveAttention(int numHiddens, float dropout) {
        W_k = Linear.builder().setUnits(numHiddens).optBias(false).build();
        addChildBlock("W_k", W_k);

        W_q = Linear.builder().setUnits(numHiddens).optBias(false).build();
        addChildBlock("W_q", W_q);

        W_v = Linear.builder().setUnits(1).optBias(false).build();
        addChildBlock("W_v", W_v);

        this.dropout = Dropout.builder().optRate(dropout).build();
        addChildBlock("dropout", this.dropout);
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore ps,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        // Shape of the output `queries` and `attentionWeights`:
        // (no. of queries, no. of key-value pairs)
        NDArray queries = inputs.get(0);
        NDArray keys = inputs.get(1);
        NDArray values = inputs.get(2);
        NDArray validLens = inputs.get(3);

        queries = W_q.forward(ps, new NDList(queries), training, params).head();
        keys = W_k.forward(ps, new NDList(keys), training, params).head();
        // After dimension expansion, shape of `queries`: (`batchSize`, no. of
        // queries, 1, `numHiddens`) and shape of `keys`: (`batchSize`, 1,
        // no. of key-value pairs, `numHiddens`). Sum them up with
        // broadcasting
        NDArray features = queries.expandDims(2).add(keys.expandDims(1));
        features = features.tanh();
        // There is only one output of `this.W_v`, so we remove the last
        // one-dimensional entry from the shape. Shape of `scores`:
        // (`batchSize`, no. of queries, no. of key-value pairs)
        NDArray result = W_v.forward(ps, new NDList(features), training, params).head();
        NDArray scores = result.squeeze(-1);
        attentionWeights = Test02.maskedSoftmax(scores, validLens);
        // Shape of `values`: (`batchSize`, no. of key-value pairs, value dimension)
        NDList list = dropout.forward(ps, new NDList(attentionWeights), training, params);
        return new NDList(list.head().batchDot(values));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void initializeChildBlocks(
            NDManager manager, DataType dataType, Shape... inputShapes) {
        W_q.initialize(manager, dataType, inputShapes[0]);
        W_k.initialize(manager, dataType, inputShapes[1]);
        long[] q = W_q.getOutputShapes(new Shape[]{inputShapes[0]})[0].getShape();
        long[] k = W_k.getOutputShapes(new Shape[]{inputShapes[1]})[0].getShape();
        long w = Math.max(q[q.length - 2], k[k.length - 2]);
        long h = Math.max(q[q.length - 1], k[k.length - 1]);
        long[] shape = new long[]{2, 1, w, h};
        W_v.initialize(manager, dataType, new Shape(shape));
        long[] dropoutShape = new long[shape.length - 1];
        System.arraycopy(shape, 0, dropoutShape, 0, dropoutShape.length);
        dropout.initialize(manager, dataType, new Shape(dropoutShape));
    }
}
