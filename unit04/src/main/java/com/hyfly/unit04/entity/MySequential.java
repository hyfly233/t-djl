package com.hyfly.unit04.entity;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public class MySequential extends AbstractBlock {

    private static final byte VERSION = 1;

    public MySequential() {
        super(VERSION);
    }

    public MySequential add(Block block) {
        // Here, block is an instance of a Block subclass, and we assume it has
        // a unique name. We add the child block to the children BlockList
        // with `addChildBlock()` which is defined in AbstractBlock.
        if (block != null) {
            addChildBlock(block.getClass().getSimpleName(), block);
        }
        return this;
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDList current = inputs;
        for (Block block : children.values()) {
            // BlockList guarantees that members will be traversed in the order
            // they were added
            current = block.forward(parameterStore, current, training);
        }
        return current;
    }

    @Override
    // Initializes all child blocks
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        Shape[] shapes = inputShapes;
        for (Block child : getChildren().values()) {
            child.initialize(manager, dataType, shapes);
            shapes = child.getOutputShapes(shapes);
        }
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        Shape[] current = inputs;
        for (Block block : children.values()) {
            current = block.getOutputShapes(current);
        }
        return current;
    }
}
