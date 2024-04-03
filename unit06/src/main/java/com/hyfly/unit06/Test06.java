package com.hyfly.unit06;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
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
import com.hyfly.unit06.entity.Residual;
import com.hyfly.utils.Training;
import org.apache.commons.lang3.ArrayUtils;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Test06 {

    public static void main(String[] args) throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            SequentialBlock blk = new SequentialBlock();
            blk.add(new Residual(3, false, new Shape(1, 1)));

            NDArray X = manager.randomUniform(0f, 1.0f, new Shape(4, 3, 6, 6));

            ParameterStore parameterStore = new ParameterStore(manager, true);

            blk.initialize(manager, DataType.FLOAT32, X.getShape());

            blk.forward(parameterStore, new NDList(X), false).singletonOrThrow().getShape();

            blk = new SequentialBlock();
            blk.add(new Residual(6, true, new Shape(2, 2)));

            blk.initialize(manager, DataType.FLOAT32, X.getShape());

            blk.forward(parameterStore, new NDList(X), false).singletonOrThrow().getShape();

            SequentialBlock net = new SequentialBlock();
            net.add(
                            Conv2d.builder()
                                    .setKernelShape(new Shape(7, 7))
                                    .optStride(new Shape(2, 2))
                                    .optPadding(new Shape(3, 3))
                                    .setFilters(64)
                                    .build())
                    .add(BatchNorm.builder().build())
                    .add(Activation::relu)
                    .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1))
                    );

            //
            net
                    .add(resnetBlock(64, 2, true))
                    .add(resnetBlock(128, 2, false))
                    .add(resnetBlock(256, 2, false))
                    .add(resnetBlock(512, 2, false));

            //
            net
                    .add(Pool.globalAvgPool2dBlock())
                    .add(Linear.builder().setUnits(10).build());

            //
            X = manager.randomUniform(0f, 1f, new Shape(1, 1, 224, 224));
            net.initialize(manager, DataType.FLOAT32, X.getShape());
            Shape currentShape = X.getShape();

            for (int i = 0; i < net.getChildren().size(); i++) {

                X = net.getChildren().get(i).getValue().forward(parameterStore, new NDList(X), false).singletonOrThrow();
                currentShape = X.getShape();
                System.out.println(net.getChildren().get(i).getKey() + " layer output : " + currentShape);
            }

            //
            int batchSize = 256;
            float lr = 0.05f;
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

    public static SequentialBlock resnetBlock(int numChannels, int numResiduals, boolean firstBlock) {
        SequentialBlock blk = new SequentialBlock();

        for (int i = 0; i < numResiduals; i++) {

            if (i == 0 && !firstBlock) {
                blk.add(new Residual(numChannels, true, new Shape(2, 2)));
            } else {
                blk.add(new Residual(numChannels, false, new Shape(1, 1)));
            }
        }
        return blk;
    }
}
