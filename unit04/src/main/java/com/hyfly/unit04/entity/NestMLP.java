package com.hyfly.unit04.entity;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public class NestMLP extends AbstractBlock {

    private SequentialBlock net;
    private Block dense;

    private Block test;

    public NestMLP() {
        net = new SequentialBlock();
        net.add(Linear.builder().setUnits(64).build());
        net.add(Activation.reluBlock());
        net.add(Linear.builder().setUnits(32).build());
        net.add(Activation.reluBlock());
        addChildBlock("net", net);

        dense = addChildBlock("dense", Linear.builder().setUnits(16).build());
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDList current = inputs;

        // Fully connected layer
        current = net.forward(parameterStore, current, training);
        current = dense.forward(parameterStore, current, training);
        current = new NDList(Activation.relu(current.singletonOrThrow()));
        return current;
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        Shape[] current = inputs;
        for (Block block : children.values()) {
            current = block.getOutputShapes(current);
        }
        return current;
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        Shape[] shapes = inputShapes;
        for (Block child : getChildren().values()) {
            child.initialize(manager, dataType, shapes);
            shapes = child.getOutputShapes(shapes);
        }
    }
}
