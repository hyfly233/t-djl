package com.hyfly.unit04.entity;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public class CenteredLayer extends AbstractBlock {

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDList current = inputs;
        // Subtract the mean from the input
        return new NDList(current.head().sub(current.head().mean()));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        // Output shape should be the same as input
        return inputs;
    }
}
