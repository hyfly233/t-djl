package com.hyfly.unit03;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.util.RandomUtils;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.special.Gamma;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.api.LinePlot;
import tech.tablesaw.plotly.components.Axis;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

@Slf4j
public class Test04 {

    public static void main(String[] args) throws Exception {
        int maxDegree = 20; // Maximum degree of the polynomial
        // Training and test dataset sizes
        int nTrain = 100;
        int nTest = 100;

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray trueW = manager.zeros(new Shape(maxDegree)); // Allocate lots of empty space
            NDArray tempArr = manager.create(new float[]{5f, 1.2f, -3.4f, 5.6f});

            for (int i = 0; i < tempArr.size(); i++) {
                trueW.set(new NDIndex(i), tempArr.getFloat(i));
            }

            NDArray features = manager.randomNormal(new Shape(nTrain + nTest, 1));
            features = shuffle(features);

            NDArray polyFeatures = features.pow(manager.arange(maxDegree).reshape(1, -1));

            for (int i = 0; i < maxDegree; i++) {
                polyFeatures.set(new NDIndex(":, " + i), polyFeatures.get(":, " + i).div(Gamma.gamma(i + 1)));
            }
            // NDArray factorialArr = factorial(manager.arange(maxDegree).add(1.0f).toType(DataType.FLOAT32, false)).reshape(1, -1);

            // polyFeatures = polyFeatures.div(factorialArr);
            // Shape of `labels`: (`n_train` + `n_test`,)
            NDArray labels = polyFeatures.dot(trueW);
            labels = labels.add(manager.randomNormal(0, 0.1f, labels.getShape(), DataType.FLOAT32));

            System.out.println("features: " + features.get(":2"));
            System.out.println("polyFeatures: " + polyFeatures.get(":2"));
            System.out.println("labels: " + labels.get(":2"));

            int logInterval = 20;
            int numEpochs = Integer.getInteger("MAX_EPOCH", 400);

            double[] trainLoss;
            double[] testLoss;
            double[] epochCount;

            trainLoss = new double[numEpochs / logInterval];
            testLoss = new double[numEpochs / logInterval];
            epochCount = new double[numEpochs / logInterval];

            NDArray weight = null;

            int nDegree = 4;
            train(polyFeatures.get("0:" + nTrain + ", 0:" + nDegree),
                    polyFeatures.get(nTrain + ": , 0:" + nDegree),
                    labels.get(":" + nTrain),
                    labels.get(nTrain + ":"), nDegree, numEpochs, logInterval, trainLoss, testLoss, epochCount);

            String[] lossLabel = new String[trainLoss.length + testLoss.length];

            Arrays.fill(lossLabel, 0, trainLoss.length, "train loss");
            Arrays.fill(lossLabel, trainLoss.length, trainLoss.length + testLoss.length, "test loss");

            Table data = Table.create("Data").addColumns(
                    DoubleColumn.create("epochCount", ArrayUtils.addAll(epochCount, epochCount)),
                    DoubleColumn.create("loss", ArrayUtils.addAll(trainLoss, testLoss)),
                    StringColumn.create("lossLabel", lossLabel)
            );
            Figure figure = LinePlot.create("Normal", data, "epochCount", "loss", "lossLabel");
// set Y axis to log scale
            Axis yAxis = Axis.builder()
                    .type(Axis.Type.LOG)
                    .build();
            Layout layout = Layout.builder("Normal")
                    .yAxis(yAxis)
                    .build();
            figure.setLayout(layout);
//            render(figure,"text/html");

            // 从多项式特征中选择前2个维度，即 1, x
            int nDegree2 = 2;
            trainLoss = new double[numEpochs / logInterval];
            testLoss = new double[numEpochs / logInterval];
            epochCount = new double[numEpochs / logInterval];
            train(polyFeatures.get("0:" + nTrain + ", 0:" + nDegree2),
                    polyFeatures.get(nTrain + ": , 0:" + nDegree2),
                    labels.get(":" + nTrain),
                    labels.get(nTrain + ":"), nDegree2, numEpochs, logInterval, trainLoss, testLoss, epochCount);

            String[] lossLabel2 = new String[trainLoss.length + testLoss.length];

            Arrays.fill(lossLabel2, 0, trainLoss.length, "train loss");
            Arrays.fill(lossLabel2, trainLoss.length, trainLoss.length + testLoss.length, "test loss");

            Table data1 = Table.create("Data").addColumns(
                    DoubleColumn.create("epochCount", ArrayUtils.addAll(epochCount, epochCount)),
                    DoubleColumn.create("loss", ArrayUtils.addAll(trainLoss, testLoss)),
                    StringColumn.create("lossLabel", lossLabel2)
            );
            Figure figure1 = LinePlot.create("Underfitting", data1, "epochCount", "loss", "lossLabel");
// set Y axis to log scale
            Axis yAxis1 = Axis.builder()
                    .type(Axis.Type.LOG)
                    .build();
            Layout layout1 = Layout.builder("Underfitting")
                    .yAxis(yAxis1)
                    .build();
            figure1.setLayout(layout1);
//            render(figure1,"text/html");

            // 从多项式特征中选取所有维度
            numEpochs = 1500;
            logInterval = 50;

            trainLoss = new double[numEpochs / logInterval];
            testLoss = new double[numEpochs / logInterval];
            epochCount = new double[numEpochs / logInterval];

            train(polyFeatures.get("0:" + nTrain + ", 0:" + maxDegree),
                    polyFeatures.get(nTrain + ": , 0:" + maxDegree),
                    labels.get(":" + nTrain),
                    labels.get(nTrain + ":"), maxDegree, numEpochs, logInterval, trainLoss, testLoss, epochCount);

            String[] lossLabel1 = new String[trainLoss.length + testLoss.length];

            Arrays.fill(lossLabel1, 0, trainLoss.length, "train loss");
            Arrays.fill(lossLabel1, trainLoss.length, trainLoss.length + testLoss.length, "test loss");

            Table data2 = Table.create("Data").addColumns(
                    DoubleColumn.create("epochCount", ArrayUtils.addAll(epochCount, epochCount)),
                    DoubleColumn.create("loss", ArrayUtils.addAll(trainLoss, testLoss)),
                    StringColumn.create("lossLabel", lossLabel1)
            );

            Figure figure2 = LinePlot.create("Overfitting", data2, "epochCount", "loss", "lossLabel");
// set Y axis to log scale
            Axis yAxis2 = Axis.builder()
                    .type(Axis.Type.LOG)
                    .build();
            Layout layout2 = Layout.builder("Overfitting")
                    .yAxis(yAxis2)
                    .build();
            figure2.setLayout(layout2);
//            render(figure2,"text/html");
        }

    }

    public static void swap(NDArray arr, int i, int j) {
        float tmp = arr.getFloat(i);
        arr.set(new NDIndex(i), arr.getFloat(j));
        arr.set(new NDIndex(j), tmp);
    }

    public static NDArray shuffle(NDArray arr) {
        int size = (int) arr.size();

        Random rnd = RandomUtils.RANDOM;

        for (int i = Math.toIntExact(size) - 1; i > 0; --i) {
            swap(arr, i, rnd.nextInt(i));
        }
        return arr;
    }

    public static ArrayDataset loadArray(NDArray features, NDArray labels, int batchSize, boolean shuffle) {
        return new ArrayDataset.Builder()
                .setData(features) // set the features
                .optLabels(labels) // set the labels
                .setSampling(batchSize, shuffle) // set the batch size and random sampling
                .build();
    }

    public static void train(NDArray trainFeatures, NDArray testFeatures, NDArray trainLabels, NDArray testLabels, int nDegree,
                             int numEpochs, int logInterval, double[] trainLoss, double[] testLoss, double[] epochCount)
            throws IOException, TranslateException {

        Loss l2Loss = Loss.l2Loss();
        NDManager manager = NDManager.newBaseManager();
        Tracker lrt = Tracker.fixed(0.01f);
        Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();
        DefaultTrainingConfig config = new DefaultTrainingConfig(l2Loss)
                .optDevices(manager.getEngine().getDevices(1)) // single GPU
                .optOptimizer(sgd) // Optimizer (loss function)
                .addTrainingListeners(TrainingListener.Defaults.basic()); // Logging

        Model model = Model.newInstance("mlp");
        SequentialBlock net = new SequentialBlock();
        // Switch off the bias since we already catered for it in the polynomial
        // features
        Linear linearBlock = Linear.builder().optBias(false).setUnits(1).build();
        net.add(linearBlock);

        model.setBlock(net);
        Trainer trainer = model.newTrainer(config);

        int batchSize = Math.min(10, (int) trainLabels.getShape().get(0));

        ArrayDataset trainIter = loadArray(trainFeatures, trainLabels, batchSize, true);
        ArrayDataset testIter = loadArray(testFeatures, testLabels, batchSize, true);

        trainer.initialize(new Shape(1, nDegree));
        System.out.println("Start Training...");
        for (int epoch = 1; epoch <= numEpochs; epoch++) {

            // Iterate over dataset
            for (Batch batch : trainer.iterateDataset(trainIter)) {
                // Update loss and evaulator
                EasyTrain.trainBatch(trainer, batch);

                // Update parameters
                trainer.step();

                batch.close();
            }
            // reset training and validation evaluators at end of epoch

            for (Batch batch : trainer.iterateDataset(testIter)) {
                // Update loss and evaulator
                EasyTrain.validateBatch(trainer, batch);

                batch.close();
            }

            trainer.notifyListeners(listener -> listener.onEpoch(trainer));
            if (epoch % logInterval == 0) {
                epochCount[epoch / logInterval - 1] = epoch;
                trainLoss[epoch / logInterval - 1] = trainer.getTrainingResult().getEvaluations().get("train_loss");
                testLoss[epoch / logInterval - 1] = trainer.getTrainingResult().getEvaluations().get("validate_loss");
            }
        }
        System.out.println("Training complete...");
        model.close();
    }
}
