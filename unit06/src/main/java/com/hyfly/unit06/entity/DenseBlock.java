package com.hyfly.unit06.entity;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import static com.hyfly.unit06.Test07.convBlock;

public class DenseBlock extends AbstractBlock {

    private static final byte VERSION = 1;

    public SequentialBlock net = new SequentialBlock();

    public DenseBlock(int numConvs, int numChannels) {
        super(VERSION);
        for (int i = 0; i < numConvs; i++) {
            net.add(addChildBlock("denseBlock" + i, convBlock(numChannels)));
        }
    }

    @Override
    public String toString() {
        return "DenseBlock()";
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList X,
            boolean training,
            PairList<String, Object> params) {
        NDArray Y;
        for (Block block : net.getChildren().values()) {
            Y = block.forward(parameterStore, X, training).singletonOrThrow();
            X = new NDList(NDArrays.concat(new NDList(X.singletonOrThrow(), Y), 1));
        }
        return X;
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        Shape[] shapesX = inputs;
        for (Block block : net.getChildren().values()) {
            Shape[] shapesY = block.getOutputShapes(shapesX);
            shapesX[0] = new Shape(
                    shapesX[0].get(0),
                    shapesY[0].get(1) + shapesX[0].get(1),
                    shapesX[0].get(2),
                    shapesX[0].get(3)
            );
        }
        return shapesX;
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        Shape shapesX = inputShapes[0];
        for (Block block : this.net.getChildren().values()) {
            block.initialize(manager, DataType.FLOAT32, shapesX);
            Shape[] shapesY = block.getOutputShapes(new Shape[]{shapesX});
            shapesX = new Shape(
                    shapesX.get(0),
                    shapesY[0].get(1) + shapesX.get(1),
                    shapesX.get(2),
                    shapesX.get(3)
            );
        }
    }
}
