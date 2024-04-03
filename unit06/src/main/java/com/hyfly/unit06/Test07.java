package com.hyfly.unit06;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.Pair;
import com.hyfly.unit06.entity.DenseBlock;
import com.hyfly.utils.Training;
import org.apache.commons.lang3.ArrayUtils;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Test07 {

    public static void main(String[] args) throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            SequentialBlock block = new SequentialBlock().add(new DenseBlock(2, 10));

            NDArray X = manager.randomUniform(0f, 1.0f, new Shape(4, 3, 8, 8));

            block.initialize(manager, DataType.FLOAT32, X.getShape());

            ParameterStore parameterStore = new ParameterStore(manager, true);

            Shape[] currentShape = new Shape[]{X.getShape()};
            for (Block child : block.getChildren().values()) {
                currentShape = child.getOutputShapes(currentShape);
            }

            Shape shape = currentShape[0];

            block = transitionBlock(10);

            block.initialize(manager, DataType.FLOAT32, currentShape);

            for (Pair<String, Block> pair : block.getChildren()) {
                currentShape = pair.getValue().getOutputShapes(currentShape);
            }

            shape = currentShape[0];

            //
            SequentialBlock net = new SequentialBlock()
                    .add(Conv2d.builder()
                            .setFilters(64)
                            .setKernelShape(new Shape(7, 7))
                            .optStride(new Shape(2, 2))
                            .optPadding(new Shape(3, 3))
                            .build())
                    .add(BatchNorm.builder().build())
                    .add(Activation::relu)
                    .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1)));

            //
            int numChannels = 64;
            int growthRate = 32;

            int[] numConvsInDenseBlocks = new int[]{4, 4, 4, 4};

            for (int index = 0; index < numConvsInDenseBlocks.length; index++) {
                int numConvs = numConvsInDenseBlocks[index];
                net.add(new DenseBlock(numConvs, growthRate));

                numChannels += (numConvs * growthRate);

                if (index != (numConvsInDenseBlocks.length - 1)) {
                    numChannels = (numChannels / 2);
                    net.add(transitionBlock(numChannels));
                }
            }

            //
            net
                    .add(BatchNorm.builder().build())
                    .add(Activation::relu)
                    .add(Pool.globalAvgPool2dBlock())
                    .add(Linear.builder().setUnits(10).build());

            //
            int batchSize = 256;
            float lr = 0.1f;
            int numEpochs = Integer.getInteger("MAX_EPOCH", 10);

            double[] trainLoss;
            double[] testAccuracy;
            double[] epochCount;
            double[] trainAccuracy;

            epochCount = new double[numEpochs];

            for (int i = 0; i < epochCount.length; i++) {
                epochCount[i] = (i + 1);
            }

            FashionMnist trainIter = FashionMnist.builder()
                    .addTransform(new Resize(96))
                    .addTransform(new ToTensor())
                    .optUsage(Dataset.Usage.TRAIN)
                    .setSampling(batchSize, true)
                    .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                    .build();

            FashionMnist testIter = FashionMnist.builder()
                    .addTransform(new Resize(96))
                    .addTransform(new ToTensor())
                    .optUsage(Dataset.Usage.TEST)
                    .setSampling(batchSize, true)
                    .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                    .build();

            trainIter.prepare();
            testIter.prepare();

            Model model = Model.newInstance("cnn");
            model.setBlock(net);

            Loss loss = Loss.softmaxCrossEntropyLoss();

            Tracker lrt = Tracker.fixed(lr);
            Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

            DefaultTrainingConfig config = new DefaultTrainingConfig(loss).optOptimizer(sgd) // Optimizer (loss function)
                    .addEvaluator(new Accuracy()) // Model Accuracy
                    .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

            Trainer trainer = model.newTrainer(config);
            trainer.initialize(new Shape(1, 1, 96, 96));

            Map<String, double[]> evaluatorMetrics = new HashMap<>();
            double avgTrainTimePerEpoch = Training.trainingChapter6(trainIter, testIter, numEpochs, trainer, evaluatorMetrics);

            //
            trainLoss = evaluatorMetrics.get("train_epoch_SoftmaxCrossEntropyLoss");
            trainAccuracy = evaluatorMetrics.get("train_epoch_Accuracy");
            testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy");

            System.out.printf("loss %.3f,", trainLoss[numEpochs - 1]);
            System.out.printf(" train acc %.3f,", trainAccuracy[numEpochs - 1]);
            System.out.printf(" test acc %.3f\n", testAccuracy[numEpochs - 1]);
            System.out.printf("%.1f examples/sec", trainIter.size() / (avgTrainTimePerEpoch / Math.pow(10, 9)));
            System.out.println();

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

//            render(LinePlot.create("", data, "epoch", "metrics", "lossLabel"),"text/html");
        }
    }

    public static SequentialBlock convBlock(int numChannels) {

        SequentialBlock block = new SequentialBlock()
                .add(BatchNorm.builder().build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setFilters(numChannels)
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .optStride(new Shape(1, 1))
                        .build()
                );

        return block;
    }

    public static SequentialBlock transitionBlock(int numChannels) {
        SequentialBlock blk = new SequentialBlock()
                .add(BatchNorm.builder().build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setFilters(numChannels)
                        .setKernelShape(new Shape(1, 1))
                        .optStride(new Shape(1, 1))
                        .build()
                )
                .add(Pool.avgPool2dBlock(new Shape(2, 2), new Shape(2, 2)));

        return blk;
    }
}
