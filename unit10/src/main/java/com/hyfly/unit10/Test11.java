package com.hyfly.unit10;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.CosineTracker;
import ai.djl.training.tracker.MultiFactorTracker;
import ai.djl.training.tracker.Tracker;
import ai.djl.training.tracker.WarmUpTracker;
import ai.djl.translate.TranslateException;
import com.hyfly.unit10.entity.CosineWarmupTracker;
import com.hyfly.unit10.entity.DemoCosineTracker;
import com.hyfly.unit10.entity.DemoFactorTracker;
import com.hyfly.unit10.entity.SquareRootTracker;
import org.apache.commons.lang3.ArrayUtils;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.IntColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.api.LinePlot;
import tech.tablesaw.plotly.components.Figure;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Test11 {

    public static void main(String[] args) throws Exception {
        SequentialBlock net = new SequentialBlock();

        net.add(Conv2d.builder()
                .setKernelShape(new Shape(5, 5))
                .optPadding(new Shape(2, 2))
                .setFilters(1)
                .build());
        net.add(Activation.reluBlock());
        net.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));
        net.add(Conv2d.builder()
                .setKernelShape(new Shape(5, 5))
                .setFilters(1)
                .build());
        net.add(Blocks.batchFlattenBlock());
        net.add(Activation.reluBlock());
        net.add(Linear.builder().setUnits(120).build());
        net.add(Activation.reluBlock());
        net.add(Linear.builder().setUnits(84).build());
        net.add(Activation.reluBlock());
        net.add(Linear.builder().setUnits(10).build());

        int batchSize = 256;
        RandomAccessDataset trainDataset = FashionMnist.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, false)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();

        RandomAccessDataset testDataset = FashionMnist.builder()
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, false)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();

        double[] trainLoss = new double[0];
        double[] testAccuracy = new double[0];
        double[] epochCount = new double[0];
        double[] trainAccuracy = new double[0];

        float lr = 0.3f;
        int numEpochs = Integer.getInteger("MAX_EPOCH", 10);

        Model model = Model.newInstance("Modern LeNet");
        model.setBlock(net);

        Loss loss = Loss.softmaxCrossEntropyLoss();
        Tracker lrt = Tracker.fixed(lr);
        Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

        DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                .optOptimizer(sgd) // Optimizer
                .addEvaluator(new Accuracy()) // Model Accuracy
                .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

        Trainer trainer = model.newTrainer(config);
        trainer.initialize(new Shape(1, 1, 28, 28));

        train(trainDataset, testDataset, numEpochs, trainer, trainLoss, testAccuracy, epochCount, trainAccuracy);

        //
        plotMetrics(trainLoss, testAccuracy, epochCount, trainAccuracy);

        //
        SquareRootTracker tracker = new SquareRootTracker();

        int[] epochs = new int[numEpochs];
        float[] learningRates = new float[numEpochs];
        for (int i = 0; i < numEpochs; i++) {
            epochs[i] = i;
            learningRates[i] = tracker.getNewLearningRate(i);
        }

        plotLearningRate(epochs, learningRates, trainLoss, testAccuracy, trainAccuracy);

        //
        DemoFactorTracker tracker1 = new DemoFactorTracker(0.9f, (float) 1e-2, 2);

        numEpochs = 50;
        epochs = new int[numEpochs];
        learningRates = new float[numEpochs];
        for (int i = 0; i < numEpochs; i++) {
            epochs[i] = i;
            learningRates[i] = tracker1.getNewLearningRate(lr, i);
        }

        plotLearningRate(epochs, learningRates, trainLoss, testAccuracy, trainAccuracy);

        //
        MultiFactorTracker tracker2 = Tracker.multiFactor()
                .setSteps(new int[]{5, 30})
                .optFactor(0.5f)
                .setBaseValue(0.5f)
                .build();

        numEpochs = 10;
        epochs = new int[numEpochs];
        learningRates = new float[numEpochs];
        for (int i = 0; i < numEpochs; i++) {
            epochs[i] = i;
            learningRates[i] = tracker2.getNewValue(i);
        }

        plotLearningRate(epochs, learningRates, trainLoss, testAccuracy, trainAccuracy);

        //
        numEpochs = Integer.getInteger("MAX_EPOCH", 10);

        model = Model.newInstance("Modern LeNet");
        model.setBlock(net);

        loss = Loss.softmaxCrossEntropyLoss();
        sgd = Optimizer.sgd().setLearningRateTracker(tracker2).build();

        config = new DefaultTrainingConfig(loss)
                .optOptimizer(sgd) // Optimizer
                .addEvaluator(new Accuracy()) // Model Accuracy
                .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

        Trainer trainer3 = model.newTrainer(config);
        trainer3.initialize(new Shape(1, 1, 28, 28));

        train(trainDataset, testDataset, numEpochs, trainer3, trainLoss, testAccuracy, epochCount, trainAccuracy);
        plotMetrics(trainLoss, testAccuracy, epochCount, trainAccuracy);

        //
        String[] lossLabel = new String[trainLoss.length + testAccuracy.length + trainAccuracy.length];

        Arrays.fill(lossLabel, 0, trainLoss.length, "train loss");
        Arrays.fill(lossLabel, trainAccuracy.length, trainLoss.length + trainAccuracy.length, "train acc");
        Arrays.fill(lossLabel, trainLoss.length + trainAccuracy.length,
                trainLoss.length + testAccuracy.length + trainAccuracy.length, "test acc");

        Table data = Table.create("Data").addColumns(
                DoubleColumn.create("epoch", ArrayUtils.addAll(epochCount, ArrayUtils.addAll(epochCount, epochCount))),
                DoubleColumn.create("metrics", ArrayUtils.addAll(trainLoss, ArrayUtils.addAll(trainAccuracy, testAccuracy))),
                StringColumn.create("lossLabel", lossLabel)
        );

        LinePlot.create("", data, "epoch", "metrics", "lossLabel");

        //
        DemoCosineTracker tracker4 = new DemoCosineTracker(0.5f, 0.01f, 20);

        epochs = new int[numEpochs];
        learningRates = new float[numEpochs];
        for (int i = 0; i < numEpochs; i++) {
            epochs[i] = i;
            learningRates[i] = tracker4.getNewLearningRate(i);
        }

        plotLearningRate(epochs, learningRates, trainLoss, testAccuracy, trainAccuracy);

        //
        CosineTracker cosineTracker = Tracker.cosine()
                .setBaseValue(0.5f)
                .optFinalValue(0.01f)
                .setMaxUpdates(20)
                .build();

        loss = Loss.softmaxCrossEntropyLoss();
        sgd = Optimizer.sgd().setLearningRateTracker(cosineTracker).build();

        config = new DefaultTrainingConfig(loss)
                .optOptimizer(sgd) // Optimizer
                .addEvaluator(new Accuracy()) // Model Accuracy
                .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

        trainer = model.newTrainer(config);
        trainer.initialize(new Shape(1, 1, 28, 28));

        train(trainDataset, testDataset, numEpochs, trainer, trainLoss, testAccuracy, epochCount, trainAccuracy);

        //
        CosineWarmupTracker tracker5 = new CosineWarmupTracker(0.5f, 0.01f, 20, 5);

        epochs = new int[numEpochs];
        learningRates = new float[numEpochs];
        for (int i = 0; i < numEpochs; i++) {
            epochs[i] = i;
            learningRates[i] = tracker5.getNewLearningRate(i);
        }

        plotLearningRate(epochs, learningRates, trainLoss, testAccuracy, trainAccuracy);

        //
        CosineTracker cosineTracker1 = Tracker.cosine()
                .setBaseValue(0.5f)
                .optFinalValue(0.01f)
                .setMaxUpdates(15)
                .build();

        WarmUpTracker warmupCosine = Tracker.warmUp()
                .optWarmUpSteps(5)
                .setMainTracker(cosineTracker1)
                .build();

        loss = Loss.softmaxCrossEntropyLoss();
        sgd = Optimizer.sgd().setLearningRateTracker(warmupCosine).build();

        config = new DefaultTrainingConfig(loss)
                .optOptimizer(sgd) // Optimizer
                .addEvaluator(new Accuracy()) // Model Accuracy
                .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

        trainer = model.newTrainer(config);
        trainer.initialize(new Shape(1, 1, 28, 28));

        train(trainDataset, testDataset, numEpochs, trainer, trainLoss, testAccuracy, epochCount, trainAccuracy);
        plotMetrics(trainLoss, testAccuracy, epochCount, trainAccuracy);

        //

    }

    public static void train(RandomAccessDataset trainIter, RandomAccessDataset testIter,
                             int numEpochs, Trainer trainer, double[] trainLoss,
                             double[] testAccuracy,
                             double[] epochCount,
                             double[] trainAccuracy) throws IOException, TranslateException {
        epochCount = new double[numEpochs];

        for (int i = 0; i < epochCount.length; i++) {
            epochCount[i] = (i + 1);
        }

        double avgTrainTimePerEpoch = 0;
        Map<String, double[]> evaluatorMetrics = new HashMap<>();

        trainer.setMetrics(new Metrics());

        EasyTrain.fit(trainer, numEpochs, trainIter, testIter);

        Metrics metrics = trainer.getMetrics();

        trainer.getEvaluators().stream()
                .forEach(evaluator -> {
                    evaluatorMetrics.put("train_epoch_" + evaluator.getName(), metrics.getMetric("train_epoch_" + evaluator.getName()).stream()
                            .mapToDouble(x -> x.getValue().doubleValue()).toArray());
                    evaluatorMetrics.put("validate_epoch_" + evaluator.getName(), metrics.getMetric("validate_epoch_" + evaluator.getName()).stream()
                            .mapToDouble(x -> x.getValue().doubleValue()).toArray());
                });

        avgTrainTimePerEpoch = metrics.mean("epoch");

        trainLoss = evaluatorMetrics.get("train_epoch_SoftmaxCrossEntropyLoss");
        trainAccuracy = evaluatorMetrics.get("train_epoch_Accuracy");
        testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy");

        System.out.printf("loss %.3f,", trainLoss[numEpochs - 1]);
        System.out.printf(" train acc %.3f,", trainAccuracy[numEpochs - 1]);
        System.out.printf(" test acc %.3f\n", testAccuracy[numEpochs - 1]);
        System.out.printf("%.1f examples/sec \n", trainIter.size() / (avgTrainTimePerEpoch / Math.pow(10, 9)));
    }

    public static void plotMetrics(double[] trainLoss,
                                   double[] testAccuracy,
                                   double[] epochCount,
                                   double[] trainAccuracy) {
        String[] lossLabel = new String[trainLoss.length + testAccuracy.length + trainAccuracy.length];

        Arrays.fill(lossLabel, 0, trainLoss.length, "train loss");
        Arrays.fill(lossLabel, trainAccuracy.length, trainLoss.length + trainAccuracy.length, "train acc");
        Arrays.fill(lossLabel, trainLoss.length + trainAccuracy.length,
                trainLoss.length + testAccuracy.length + trainAccuracy.length, "test acc");

        Table data = Table.create("Data").addColumns(
                DoubleColumn.create("epoch", ArrayUtils.addAll(epochCount, ArrayUtils.addAll(epochCount, epochCount))),
                DoubleColumn.create("metrics", ArrayUtils.addAll(trainLoss, ArrayUtils.addAll(trainAccuracy, testAccuracy))),
                StringColumn.create("lossLabel", lossLabel)
        );

//        display(LinePlot.create("", data, "epoch", "metrics", "lossLabel"));
    }

    public static Figure plotLearningRate(int[] epochs, float[] learningRates, double[] trainLoss,
                                          double[] testAccuracy,

                                          double[] trainAccuracy) {

        String[] lossLabel = new String[trainLoss.length + testAccuracy.length + trainAccuracy.length];

        Arrays.fill(lossLabel, 0, trainLoss.length, "train loss");
        Arrays.fill(lossLabel, trainAccuracy.length, trainLoss.length + trainAccuracy.length, "train acc");
        Arrays.fill(lossLabel, trainLoss.length + trainAccuracy.length,
                trainLoss.length + testAccuracy.length + trainAccuracy.length, "test acc");

        Table data = Table.create("Data").addColumns(
                IntColumn.create("epoch", epochs),
                DoubleColumn.create("learning rate", learningRates)
        );

        return LinePlot.create("Learning Rate vs. Epoch", data, "epoch", "learning rate");
    }
}
