package com.hyfly.unit03;

import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.loss.Loss;
import com.hyfly.utils.Training;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;

import java.util.Arrays;

@Slf4j
public class Test02 {

    public static void main(String[] args) throws Exception {
        int batchSize = 256;

        FashionMnist trainIter = FashionMnist.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, true)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();

        FashionMnist testIter = FashionMnist.builder()
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, true)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();

        trainIter.prepare();
        testIter.prepare();

        int numInputs = 784;
        int numOutputs = 10;
        int numHiddens = 256;

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray W1 = manager.randomNormal(0, 0.01f, new Shape(numInputs, numHiddens), DataType.FLOAT32);
            NDArray b1 = manager.zeros(new Shape(numHiddens));
            NDArray W2 = manager.randomNormal(0, 0.01f, new Shape(numHiddens, numOutputs), DataType.FLOAT32);
            NDArray b2 = manager.zeros(new Shape(numOutputs));

            NDList params = new NDList(W1, b1, W2, b2);

            for (NDArray param : params) {
                param.setRequiresGradient(true);
            }

            Loss loss = Loss.softmaxCrossEntropyLoss();

            int numEpochs = Integer.getInteger("MAX_EPOCH", 10);
            float lr = 0.5f;

            double[] trainLoss = new double[numEpochs];
            double[] trainAccuracy = new double[numEpochs];
            double[] testAccuracy = new double[numEpochs];
            double[] epochCount = new double[numEpochs];

            float epochLoss = 0f;
            float accuracyVal = 0f;

            for (int epoch = 1; epoch <= numEpochs; epoch++) {

                System.out.print("Running epoch " + epoch + "...... ");
                // Iterate over dataset
                for (Batch batch : trainIter.getData(manager)) {

                    NDArray X = batch.getData().head();
                    NDArray y = batch.getLabels().head();

                    try(GradientCollector gc = Engine.getInstance().newGradientCollector()) {
//                        NDArray yHat = net(X); // net function call

//                        NDArray lossValue = loss.evaluate(new NDList(y), new NDList(yHat));
//                        NDArray l = lossValue.mul(batchSize);

//                        accuracyVal += Training.accuracy(yHat, y);
//                        epochLoss += l.sum().getFloat();

//                        gc.backward(l); // gradient calculation
                    }

                    batch.close();
//                    Training.sgd(params, lr, batchSize); // updater
                }

                trainLoss[epoch-1] = epochLoss/trainIter.size();
                trainAccuracy[epoch-1] = accuracyVal/trainIter.size();

                epochLoss = 0f;
                accuracyVal = 0f;
                // testing now
                for (Batch batch : testIter.getData(manager)) {

                    NDArray X = batch.getData().head();
                    NDArray y = batch.getLabels().head();

//                    NDArray yHat = net(X); // net function call
//                    accuracyVal += Training.accuracy(yHat, y);
                }

                testAccuracy[epoch-1] = accuracyVal/testIter.size();
                epochCount[epoch-1] = epoch;
                accuracyVal = 0f;
                System.out.println("Finished epoch " + epoch);
            }

            System.out.println("Finished training!");

            String[] lossLabel = new String[trainLoss.length + testAccuracy.length + trainAccuracy.length];

            Arrays.fill(lossLabel, 0, trainLoss.length, "train loss");
            Arrays.fill(lossLabel, trainAccuracy.length, trainLoss.length + trainAccuracy.length, "train acc");
            Arrays.fill(lossLabel, trainLoss.length + trainAccuracy.length,
                    trainLoss.length + testAccuracy.length + trainAccuracy.length, "test acc");

            Table data = Table.create("Data").addColumns(
                    DoubleColumn.create("epochCount", ArrayUtils.addAll(epochCount, ArrayUtils.addAll(epochCount, epochCount))),
                    DoubleColumn.create("loss", ArrayUtils.addAll(trainLoss, ArrayUtils.addAll(trainAccuracy, testAccuracy))),
                    StringColumn.create("lossLabel", lossLabel)
            );

//            render(LinePlot.create("", data, "epochCount", "loss", "lossLabel"),"text/html");
        }


    }

    public static NDArray relu(NDArray X) {
        return X.maximum(0f);
    }
}
