package com.hyfly.unit06;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.hyfly.utils.Training;
import org.apache.commons.lang3.ArrayUtils;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class Test04 {

    public static void main(String[] args) throws Exception {
        SequentialBlock block1 = new SequentialBlock();
        block1
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(7, 7))
                        .optPadding(new Shape(3, 3))
                        .optStride(new Shape(2, 2))
                        .setFilters(64)
                        .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1)));

        SequentialBlock block2 = new SequentialBlock();
        block2
                .add(Conv2d.builder()
                        .setFilters(64)
                        .setKernelShape(new Shape(1, 1))
                        .build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setFilters(192)
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1)));

        SequentialBlock block3 = new SequentialBlock();
        block3
                .add(inceptionBlock(64, new int[]{96, 128}, new int[]{16, 32}, 32))
                .add(inceptionBlock(128, new int[]{128, 192}, new int[]{32, 96}, 64))
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1)));

        SequentialBlock block4 = new SequentialBlock();
        block4
                .add(inceptionBlock(192, new int[]{96, 208}, new int[]{16, 48}, 64))
                .add(inceptionBlock(160, new int[]{112, 224}, new int[]{24, 64}, 64))
                .add(inceptionBlock(128, new int[]{128, 256}, new int[]{24, 64}, 64))
                .add(inceptionBlock(112, new int[]{144, 288}, new int[]{32, 64}, 64))
                .add(inceptionBlock(256, new int[]{160, 320}, new int[]{32, 128}, 128))
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1)));

        SequentialBlock block5 = new SequentialBlock();
        block5
                .add(inceptionBlock(256, new int[]{160, 320}, new int[]{32, 128}, 128))
                .add(inceptionBlock(384, new int[]{192, 384}, new int[]{48, 128}, 128))
                .add(Pool.globalAvgPool2dBlock());

        SequentialBlock block = new SequentialBlock();
        block = block.addAll(block1, block2, block3, block4, block5, Linear.builder().setUnits(10).build());

        try (NDManager manager = NDManager.newBaseManager()) {


            float lr = 0.1f;
            Model model = Model.newInstance("cnn");
            model.setBlock(block);

            Loss loss = Loss.softmaxCrossEntropyLoss();

            Tracker lrt = Tracker.fixed(lr);
            Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

            DefaultTrainingConfig config = new DefaultTrainingConfig(loss).optOptimizer(sgd) // Optimizer (loss function)
                    .optDevices(Engine.getInstance().getDevices(1)) // single GPU
                    .addEvaluator(new Accuracy()) // Model Accuracy
                    .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

            Trainer trainer = model.newTrainer(config);

            NDArray X = manager.randomUniform(0f, 1.0f, new Shape(1, 1, 96, 96));
            trainer.initialize(X.getShape());
            Shape currentShape = X.getShape();

            for (int i = 0; i < block.getChildren().size(); i++) {
                Shape[] newShape = block.getChildren().get(i).getValue().getOutputShapes(new Shape[]{currentShape});
                currentShape = newShape[0];
                System.out.println(block.getChildren().get(i).getKey() + i + " layer output : " + currentShape);
            }

            int batchSize = 128;
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

    // c1 - c4 are the number of output channels for each layer in the path
    public static ParallelBlock inceptionBlock(int c1, int[] c2, int[] c3, int c4) {

        // Path 1 is a single 1 x 1 convolutional layer
        SequentialBlock p1 = new SequentialBlock().add(
                        Conv2d.builder()
                                .setFilters(c1)
                                .setKernelShape(new Shape(1, 1))
                                .build())
                .add(Activation::relu);

        // Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        // convolutional layer
        SequentialBlock p2 = new SequentialBlock().add(
                        Conv2d.builder()
                                .setFilters(c2[0])
                                .setKernelShape(new Shape(1, 1))
                                .build())
                .add(Activation::relu)
                .add(
                        Conv2d.builder()
                                .setFilters(c2[1])
                                .setKernelShape(new Shape(3, 3))
                                .optPadding(new Shape(1, 1))
                                .build())
                .add(Activation::relu);

        // Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        // convolutional layer
        SequentialBlock p3 = new SequentialBlock().add(
                        Conv2d.builder()
                                .setFilters(c3[0])
                                .setKernelShape(new Shape(1, 1))
                                .build())
                .add(Activation::relu)
                .add(
                        Conv2d.builder()
                                .setFilters(c3[1])
                                .setKernelShape(new Shape(5, 5))
                                .optPadding(new Shape(2, 2))
                                .build())
                .add(Activation::relu);

        // Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        // convolutional layer
        SequentialBlock p4 = new SequentialBlock()
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(1, 1), new Shape(1, 1)))
                .add(Conv2d.builder()
                        .setFilters(c4)
                        .setKernelShape(new Shape(1, 1))
                        .build())
                .add(Activation::relu);

        // Concatenate the outputs on the channel dimension
        return new ParallelBlock(
                list -> {
                    List<NDArray> concatenatedList = list
                            .stream()
                            .map(NDList::head)
                            .collect(Collectors.toList());

                    return new NDList(NDArrays.concat(new NDList(concatenatedList), 1));
                }, Arrays.asList(p1, p2, p3, p4));
    }
}
