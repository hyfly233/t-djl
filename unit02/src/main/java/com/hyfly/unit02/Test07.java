package com.hyfly.unit02;

import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.Dataset;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Test07 {

    public static void main(String[] args) {
        int batchSize = 256;
        boolean randomShuffle = true;

        // get training and validation dataset
        FashionMnist trainingSet = FashionMnist.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, randomShuffle)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();

        FashionMnist validationSet = FashionMnist.builder()
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, false)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();

        int numInputs = 784;
        int numOutputs = 10;

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray W = manager.randomNormal(0, 0.01f, new Shape(numInputs, numOutputs), DataType.FLOAT32);
            NDArray b = manager.zeros(new Shape(numOutputs), DataType.FLOAT32);
            NDList params = new NDList(W, b);

            NDArray X = manager.create(new int[][]{{1, 2, 3}, {4, 5, 6}});
            System.out.println(X.sum(new int[]{0}, true));
            System.out.println(X.sum(new int[]{1}, true));
            System.out.println(X.sum(new int[]{0, 1}, true));
        }

    }

    public static NDArray softmax(NDArray X) {
        NDArray Xexp = X.exp();
        NDArray partition = Xexp.sum(new int[]{1}, true);
        return Xexp.div(partition); // 这里应用了广播机制
    }
}
