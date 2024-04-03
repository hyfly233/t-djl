package com.hyfly.unit06.entity;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.*;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.Arrays;

public class Residual extends AbstractBlock {

    private static final byte VERSION = 2;

    public ParallelBlock block;

    public Residual(int numChannels, boolean use1x1Conv, Shape strideShape) {
        super(VERSION);

        SequentialBlock b1;
        SequentialBlock conv1x1;

        b1 = new SequentialBlock();

        b1.add(Conv2d.builder()
                        .setFilters(numChannels)
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .optStride(strideShape)
                        .build())
                .add(BatchNorm.builder().build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setFilters(numChannels)
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .build())
                .add(BatchNorm.builder().build());

        if (use1x1Conv) {
            conv1x1 = new SequentialBlock();
            conv1x1.add(Conv2d.builder()
                    .setFilters(numChannels)
                    .setKernelShape(new Shape(1, 1))
                    .optStride(strideShape)
                    .build());
        } else {
            conv1x1 = new SequentialBlock();
            conv1x1.add(Blocks.identityBlock());
        }

        block = addChildBlock("residualBlock", new ParallelBlock(
                list -> {
                    NDList unit = list.get(0);
                    NDList parallel = list.get(1);
                    return new NDList(
                            unit.singletonOrThrow()
                                    .add(parallel.singletonOrThrow())
                                    .getNDArrayInternal()
                                    .relu());
                },
                Arrays.asList(b1, conv1x1)));
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        return block.forward(parameterStore, inputs, training);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        Shape[] current = inputs;
        for (Block block : block.getChildren().values()) {
            current = block.getOutputShapes(current);
        }
        return current;
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        block.initialize(manager, dataType, inputShapes);
    }
}
