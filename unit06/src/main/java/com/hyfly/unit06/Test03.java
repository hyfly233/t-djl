package com.hyfly.unit06;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.norm.Dropout;
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
import java.util.Map;

public class Test03 {

    public static void main(String[] args) throws Exception {
        // setting the seed for demonstration purpose. You can remove it when you run the notebook
        Engine.getInstance().setRandomSeed(5555);

        SequentialBlock block = new SequentialBlock();

        block.add(niNBlock(96, new Shape(11, 11), new Shape(4, 4), new Shape(0, 0)))
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                .add(niNBlock(256, new Shape(5, 5), new Shape(1, 1), new Shape(2, 2)))
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                .add(niNBlock(384, new Shape(3, 3), new Shape(1, 1), new Shape(1, 1)))
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                .add(Dropout.builder().optRate(0.5f).build())
                // There are 10 label classes
                .add(niNBlock(10, new Shape(3, 3), new Shape(1, 1), new Shape(1, 1)))
                // The global average pooling layer automatically sets the window shape
                // to the height and width of the input
                .add(Pool.globalAvgPool2dBlock())
                // Transform the four-dimensional output into two-dimensional output
                // with a shape of (batch size, 10)
                .add(Blocks.batchFlattenBlock());

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

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray X = manager.randomUniform(0f, 1.0f, new Shape(1, 1, 224, 224));
            trainer.initialize(X.getShape());

            Shape currentShape = X.getShape();

            for (int i = 0; i < block.getChildren().size(); i++) {

                Shape[] newShape = block.getChildren().get(i).getValue().getOutputShapes(new Shape[]{currentShape});
                currentShape = newShape[0];
                System.out.println(block.getChildren().get(i).getKey() + " layer output : " + currentShape);
            }

            int batchSize = 128;
            int numEpochs = Integer.getInteger("MAX_EPOCH", 10);

            double[] trainLoss;
            double[] testAccuracy;
            double[] epochCount;
            double[] trainAccuracy;

            epochCount = new double[numEpochs];

            for (int i = 0; i < epochCount.length; i++) {
                epochCount[i] = i + 1;
            }

            FashionMnist trainIter = FashionMnist.builder()
                    .addTransform(new Resize(224))
                    .addTransform(new ToTensor())
                    .optUsage(Dataset.Usage.TRAIN)
                    .setSampling(batchSize, true)
                    .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                    .build();

            FashionMnist testIter = FashionMnist.builder()
                    .addTransform(new Resize(224))
                    .addTransform(new ToTensor())
                    .optUsage(Dataset.Usage.TEST)
                    .setSampling(batchSize, true)
                    .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                    .build();

            trainIter.prepare();
            testIter.prepare();

            //
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

    public static SequentialBlock niNBlock(int numChannels, Shape kernelShape,
                                           Shape strideShape, Shape paddingShape) {

        SequentialBlock tempBlock = new SequentialBlock();

        tempBlock.add(Conv2d.builder()
                        .setKernelShape(kernelShape)
                        .optStride(strideShape)
                        .optPadding(paddingShape)
                        .setFilters(numChannels)
                        .build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(1, 1))
                        .setFilters(numChannels)
                        .build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(1, 1))
                        .setFilters(numChannels)
                        .build())
                .add(Activation::relu);

        return tempBlock;
    }
}
