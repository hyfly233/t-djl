package com.hyfly.unit06.entity;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public class BatchNormBlock extends AbstractBlock {

    private NDArray movingMean;
    private NDArray movingVar;
    private Parameter gamma;
    private Parameter beta;
    private Shape shape;

    // num_features: the number of outputs for a fully-connected layer
    // or the number of output channels for a convolutional layer.
    // num_dims: 2 for a fully-connected layer and 4 for a convolutional layer.
    public BatchNormBlock(int numFeatures, int numDimensions) {
        if (numDimensions == 2) {
            shape = new Shape(1, numFeatures);
        } else {
            shape = new Shape(1, numFeatures, 1, 1);
        }
        // The scale parameter and the shift parameter involved in gradient
        // finding and iteration are initialized to 0 and 1 respectively
        gamma = addParameter(
                Parameter.builder()
                        .setName("gamma")
                        .setType(Parameter.Type.GAMMA)
                        .optShape(shape)
                        .build());

        beta = addParameter(
                Parameter.builder()
                        .setName("beta")
                        .setType(Parameter.Type.BETA)
                        .optShape(shape)
                        .build());

        // All the variables not involved in gradient finding and iteration are
        // initialized to 0. Create a base manager to maintain their values
        // throughout the entire training process
        NDManager manager = NDManager.newBaseManager();
        movingMean = manager.zeros(shape);
        movingVar = manager.zeros(shape);
    }

    public static NDList batchNormUpdate(NDArray X, NDArray gamma,
                                         NDArray beta, NDArray movingMean, NDArray movingVar,
                                         float eps, float momentum, boolean isTraining) {
        // attach moving mean and var to submanager to close intermediate computation values
        // at the end to avoid memory leak
        try (NDManager subManager = movingMean.getManager().newSubManager()) {
            movingMean.attach(subManager);
            movingVar.attach(subManager);
            NDArray xHat;
            NDArray mean;
            NDArray var;
            if (!isTraining) {
                // If it is the prediction mode, directly use the mean and variance
                // obtained from the incoming moving average
                xHat = X.sub(movingMean).div(movingVar.add(eps).sqrt());
            } else {
                if (X.getShape().dimension() == 2) {
                    // When using a fully connected layer, calculate the mean and
                    // variance on the feature dimension
                    mean = X.mean(new int[]{0}, true);
                    var = X.sub(mean).pow(2).mean(new int[]{0}, true);
                } else {
                    // When using a two-dimensional convolutional layer, calculate the
                    // mean and variance on the channel dimension (axis=1). Here we
                    // need to maintain the shape of `X`, so that the broadcast
                    // operation can be carried out later
                    mean = X.mean(new int[]{0, 2, 3}, true);
                    var = X.sub(mean).pow(2).mean(new int[]{0, 2, 3}, true);
                }
                // In training mode, the current mean and variance are used for the
                // standardization
                xHat = X.sub(mean).div(var.add(eps).sqrt());
                // Update the mean and variance of the moving average
                movingMean = movingMean.mul(momentum).add(mean.mul(1.0f - momentum));
                movingVar = movingVar.mul(momentum).add(var.mul(1.0f - momentum));
            }
            NDArray Y = xHat.mul(gamma).add(beta); // Scale and shift
            // attach moving mean and var back to original manager to keep their values
            movingMean.attach(subManager.getParentManager());
            movingVar.attach(subManager.getParentManager());
            return new NDList(Y, movingMean, movingVar);
        }
    }

    @Override
    public String toString() {
        return "BatchNormBlock()";
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDList result = batchNormUpdate(inputs.singletonOrThrow(),
                gamma.getArray(), beta.getArray(), this.movingMean, this.movingVar, 1e-12f, 0.9f, training);
        // close previous NDArray before assigning new values
        if (training) {
            this.movingMean.close();
            this.movingVar.close();
        }
        // Save the updated `movingMean` and `movingVar`
        this.movingMean = result.get(1);
        this.movingVar = result.get(2);
        return new NDList(result.get(0));
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
