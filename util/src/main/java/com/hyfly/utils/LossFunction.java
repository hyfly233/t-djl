package com.hyfly.utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.index.NDIndex;

// Cross Entropy only cares about the target class's probability
// Get the column index for each row
public class LossFunction {
    public static NDArray crossEntropy(NDArray yHat, NDArray y) {
        // Here, y is not guranteed to be of datatype int or long
        // and in our case we know its a float32.
        // We must first convert it to int or long(here we choose int)
        // before we can use it with NDIndex to "pick" indices.
        // It also takes in a boolean for returning a copy of the existing NDArray
        // but we don't want that so we pass in `false`.
        NDIndex pickIndex = new NDIndex()
                .addAllDim(Math.floorMod(-1, yHat.getShape().dimension()))
                .addPickDim(y);

        NDArray ndArray = yHat.get(pickIndex);
        return ndArray.log().neg();
    }
}
