package com.hyfly.unit06;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.hyfly.unit06.entity.BatchNormBlock;
import com.hyfly.utils.Training;
import org.apache.commons.lang3.ArrayUtils;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Test05 {

    public static void main(String[] args) throws Exception {
        SequentialBlock net = new SequentialBlock()
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(5, 5))
                                .setFilters(6).build())
                .add(new BatchNormBlock(6, 4))
                .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)))
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(5, 5))
                                .setFilters(16).build())
                .add(new BatchNormBlock(16, 4))
                .add(Activation::sigmoid)
                .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)))
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(120).build())
                .add(new BatchNormBlock(120, 2))
                .add(Activation::sigmoid)
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(84).build())
                .add(new BatchNormBlock(84, 2))
                .add(Activation::sigmoid)
                .add(Linear.builder().setUnits(10).build());

        int batchSize = 256;
        int numEpochs = Integer.getInteger("MAX_EPOCH", 10);

        double[] trainLoss;
        double[] testAccuracy;
        double[] epochCount;
        double[] trainAccuracy;

        epochCount = new double[numEpochs];

        for (int i = 0; i < epochCount.length; i++) {
            epochCount[i] = i + 1;
        }

        //
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

        //
        float lr = 1.0f;

        Loss loss = Loss.softmaxCrossEntropyLoss();

        Tracker lrt = Tracker.fixed(lr);
        Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

        DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                .optOptimizer(sgd) // Optimizer (loss function)
                .optDevices(Engine.getInstance().getDevices(1)) // single GPU
                .addEvaluator(new Accuracy()) // Model Accuracy
                .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

        Model model = Model.newInstance("batch-norm");
        model.setBlock(net);
        Trainer trainer = model.newTrainer(config);
        trainer.initialize(new Shape(1, 1, 28, 28));

        Map<String, double[]> evaluatorMetrics = new HashMap<>();
        double avgTrainTimePerEpoch = Training.trainingChapter6(trainIter, testIter, numEpochs, trainer, evaluatorMetrics);

        trainLoss = evaluatorMetrics.get("train_epoch_SoftmaxCrossEntropyLoss");
        trainAccuracy = evaluatorMetrics.get("train_epoch_Accuracy");
        testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy");

        System.out.printf("loss %.3f,", trainLoss[numEpochs - 1]);
        System.out.printf(" train acc %.3f,", trainAccuracy[numEpochs - 1]);
        System.out.printf(" test acc %.3f\n", testAccuracy[numEpochs - 1]);
        System.out.printf("%.1f examples/sec", trainIter.size() / (avgTrainTimePerEpoch / Math.pow(10, 9)));
        System.out.println();

        // Printing the value of gamma and beta in the first BatchNorm layer.
        List<Parameter> batchNormFirstParams = net.getChildren().values().get(1).getParameters().values();
        System.out.println("gamma " + batchNormFirstParams.get(0).getArray().reshape(-1));
        System.out.println("beta " + batchNormFirstParams.get(1).getArray().reshape(-1));

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

//        render(LinePlot.create("", data, "epoch", "metrics", "lossLabel"),"text/html");

        SequentialBlock block = new SequentialBlock()
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(5, 5))
                                .setFilters(6).build())
                .add(BatchNorm.builder().build())
                .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)))
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(5, 5))
                                .setFilters(16).build())
                .add(BatchNorm.builder().build())
                .add(Activation::sigmoid)
                .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)))
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(120).build())
                .add(BatchNorm.builder().build())
                .add(Activation::sigmoid)
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(84).build())
                .add(BatchNorm.builder().build())
                .add(Activation::sigmoid)
                .add(Linear.builder().setUnits(10).build());

        loss = Loss.softmaxCrossEntropyLoss();

        lrt = Tracker.fixed(1.0f);
        sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

        model = Model.newInstance("batch-norm");
        model.setBlock(block);

        config = new DefaultTrainingConfig(loss)
                .optOptimizer(sgd) // Optimizer (loss function)
                .addEvaluator(new Accuracy()) // Model Accuracy
                .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

        trainer = model.newTrainer(config);
        trainer.initialize(new Shape(1, 1, 28, 28));

        evaluatorMetrics = new HashMap<>();
        avgTrainTimePerEpoch = 0;


        avgTrainTimePerEpoch = Training.trainingChapter6(trainIter, testIter, numEpochs, trainer, evaluatorMetrics);


        trainLoss = evaluatorMetrics.get("train_epoch_SoftmaxCrossEntropyLoss");
        trainAccuracy = evaluatorMetrics.get("train_epoch_Accuracy");
        testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy");

        System.out.printf("loss %.3f,", trainLoss[numEpochs - 1]);
        System.out.printf(" train acc %.3f,", trainAccuracy[numEpochs - 1]);
        System.out.printf(" test acc %.3f\n", testAccuracy[numEpochs - 1]);
        System.out.printf("%.1f examples/sec", trainIter.size() / (avgTrainTimePerEpoch / Math.pow(10, 9)));
        System.out.println();

        //
        lossLabel = new String[trainLoss.length + testAccuracy.length + trainAccuracy.length];

        Arrays.fill(lossLabel, 0, trainLoss.length, "train loss");
        Arrays.fill(lossLabel, trainAccuracy.length, trainLoss.length + trainAccuracy.length, "train acc");
        Arrays.fill(lossLabel, trainLoss.length + trainAccuracy.length,
                trainLoss.length + testAccuracy.length + trainAccuracy.length, "test acc");

        data = Table.create("Data").addColumns(
                DoubleColumn.create("epoch", ArrayUtils.addAll(epochCount, ArrayUtils.addAll(epochCount, epochCount))),
                DoubleColumn.create("metrics", ArrayUtils.addAll(trainLoss, ArrayUtils.addAll(trainAccuracy, testAccuracy))),
                StringColumn.create("lossLabel", lossLabel)
        );

//        render(LinePlot.create("", data, "epoch", "metrics", "lossLabel"),"text/html");

    }
}
