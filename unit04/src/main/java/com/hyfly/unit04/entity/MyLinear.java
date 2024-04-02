package com.hyfly.unit04.entity;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public class MyLinear extends AbstractBlock {

    private Parameter weight;
    private Parameter bias;

    private int inUnits;
    private int outUnits;

    // outUnits: the number of outputs in this layer
    // inUnits: the number of inputs in this layer
    public MyLinear(int outUnits, int inUnits) {
        this.inUnits = inUnits;
        this.outUnits = outUnits;
        weight = addParameter(
                Parameter.builder()
                        .setName("weight")
                        .setType(Parameter.Type.WEIGHT)
                        .optShape(new Shape(inUnits, outUnits))
                        .build());
        bias = addParameter(
                Parameter.builder()
                        .setName("bias")
                        .setType(Parameter.Type.BIAS)
                        .optShape(new Shape(outUnits))
                        .build());
    }

    // Applies linear transformation
    public static NDArray linear(NDArray input, NDArray weight, NDArray bias) {
        return input.dot(weight).add(bias);
    }

    // Applies relu transformation
    public static NDList relu(NDArray input) {
        return new NDList(Activation.relu(input));
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray input = inputs.singletonOrThrow();
        Device device = input.getDevice();
        // Since we added the parameter, we can now access it from the parameter store
        NDArray weightArr = parameterStore.getValue(weight, device, false);
        NDArray biasArr = parameterStore.getValue(bias, device, false);
        return relu(linear(input, weightArr, biasArr));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        return new Shape[]{new Shape(outUnits, inUnits)};
    }
}
