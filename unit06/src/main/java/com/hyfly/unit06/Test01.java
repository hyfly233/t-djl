package com.hyfly.unit06;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
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


public class Test01 {

    public static void main(String[] args) throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            SequentialBlock block = new SequentialBlock();
            // Here, we use a larger 11 x 11 window to capture objects. At the same time,
            // we use a stride of 4 to greatly reduce the height and width of the output.
            //Here, the number of output channels is much larger than that in LeNet
            block
                    .add(Conv2d.builder()
                            .setKernelShape(new Shape(11, 11))
                            .optStride(new Shape(4, 4))
                            .setFilters(96).build())
                    .add(Activation::relu)
                    .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                    // Make the convolution window smaller, set padding to 2 for consistent
                    // height and width across the input and output, and increase the
                    // number of output channels
                    .add(Conv2d.builder()
                            .setKernelShape(new Shape(5, 5))
                            .optPadding(new Shape(2, 2))
                            .setFilters(256).build())
                    .add(Activation::relu)
                    .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                    // Use three successive convolutional layers and a smaller convolution
                    // window. Except for the final convolutional layer, the number of
                    // output channels is further increased. Pooling layers are not used to
                    // reduce the height and width of input after the first two
                    // convolutional layers
                    .add(Conv2d.builder()
                            .setKernelShape(new Shape(3, 3))
                            .optPadding(new Shape(1, 1))
                            .setFilters(384).build())
                    .add(Activation::relu)
                    .add(Conv2d.builder()
                            .setKernelShape(new Shape(3, 3))
                            .optPadding(new Shape(1, 1))
                            .setFilters(384).build())
                    .add(Activation::relu)
                    .add(Conv2d.builder()
                            .setKernelShape(new Shape(3, 3))
                            .optPadding(new Shape(1, 1))
                            .setFilters(256).build())
                    .add(Activation::relu)
                    .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                    // Here, the number of outputs of the fully connected layer is several
                    // times larger than that in LeNet. Use the dropout layer to mitigate
                    // overfitting
                    .add(Blocks.batchFlattenBlock())
                    .add(Linear
                            .builder()
                            .setUnits(4096)
                            .build())
                    .add(Activation::relu)
                    .add(Dropout
                            .builder()
                            .optRate(0.5f)
                            .build())
                    .add(Linear
                            .builder()
                            .setUnits(4096)
                            .build())
                    .add(Activation::relu)
                    .add(Dropout
                            .builder()
                            .optRate(0.5f)
                            .build())
                    // Output layer. Since we are using Fashion-MNIST, the number of
                    // classes is 10, instead of 1000 as in the paper
                    .add(Linear.builder().setUnits(10).build());

            //
            float lr = 0.01f;

            Model model = Model.newInstance("cnn");
            model.setBlock(block);

            Loss loss = Loss.softmaxCrossEntropyLoss();

            Tracker lrt = Tracker.fixed(lr);
            Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

            DefaultTrainingConfig config = new DefaultTrainingConfig(loss).optOptimizer(sgd) // Optimizer (loss function)
                    .addEvaluator(new Accuracy()) // Model Accuracy
                    .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

            Trainer trainer = model.newTrainer(config);

            NDArray X = manager.randomUniform(0f, 1.0f, new Shape(1, 1, 224, 224));
            trainer.initialize(X.getShape());

            Shape currentShape = X.getShape();

            for (int i = 0; i < block.getChildren().size(); i++) {
                Shape[] newShape = block.getChildren().get(i).getValue().getOutputShapes(new Shape[]{currentShape});
                currentShape = newShape[0];
                System.out.println(block.getChildren().get(i).getKey() + " layer output : " + currentShape);
            }

            //
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

            trainLoss = evaluatorMetrics.get("train_epoch_SoftmaxCrossEntropyLoss");
            trainAccuracy = evaluatorMetrics.get("train_epoch_Accuracy");
            testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy");

            System.out.printf("loss %.3f,", trainLoss[numEpochs - 1]);
            System.out.printf(" train acc %.3f,", trainAccuracy[numEpochs - 1]);
            System.out.printf(" test acc %.3f\n", testAccuracy[numEpochs - 1]);
            System.out.printf("%.1f examples/sec", trainIter.size() / (avgTrainTimePerEpoch / Math.pow(10, 9)));
            System.out.println();

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
}
