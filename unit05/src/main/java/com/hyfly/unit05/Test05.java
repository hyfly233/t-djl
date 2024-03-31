package com.hyfly.unit05;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
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
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@Slf4j
public class Test05 {

    public static void main(String[] args) throws Exception {
        Engine.getInstance().setRandomSeed(1111);

        try (NDManager manager = NDManager.newBaseManager()) {
            SequentialBlock block = new SequentialBlock();

            block.add(Conv2d.builder()
                            .setKernelShape(new Shape(5, 5))
                            .optPadding(new Shape(2, 2))
                            .optBias(false)
                            .setFilters(6)
                            .build())
                    .add(Activation::sigmoid)
                    .add(Pool.avgPool2dBlock(new Shape(5, 5), new Shape(2, 2), new Shape(2, 2)))
                    .add(Conv2d.builder()
                            .setKernelShape(new Shape(5, 5))
                            .setFilters(16).build())
                    .add(Activation::sigmoid)
                    .add(Pool.avgPool2dBlock(new Shape(5, 5), new Shape(2, 2), new Shape(2, 2)))
                    // Blocks.batchFlattenBlock() 将转换形状的输入（批次大小、通道、高度、宽度）
                    // 输入形状（批量大小,通道*高度*宽度）
                    .add(Blocks.batchFlattenBlock())
                    .add(Linear
                            .builder()
                            .setUnits(120)
                            .build())
                    .add(Activation::sigmoid)
                    .add(Linear
                            .builder()
                            .setUnits(84)
                            .build())
                    .add(Activation::sigmoid)
                    .add(Linear
                            .builder()
                            .setUnits(10)
                            .build());

            //
            float lr = 0.9f;
            Model model = Model.newInstance("cnn");
            model.setBlock(block);

            Loss loss = Loss.softmaxCrossEntropyLoss();

            Tracker lrt = Tracker.fixed(lr);
            Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

            DefaultTrainingConfig config = new DefaultTrainingConfig(loss).optOptimizer(sgd) // 优化器（损失函数）
                    .optDevices(Engine.getInstance().getDevices(1)) // 单个GPU
                    .addEvaluator(new Accuracy()) // 模型精度
                    .addTrainingListeners(TrainingListener.Defaults.basic());

            Trainer trainer = model.newTrainer(config);

            NDArray X = manager.randomUniform(0f, 1.0f, new Shape(1, 1, 28, 28));
            trainer.initialize(X.getShape());

            Shape currentShape = X.getShape();

            for (int i = 0; i < block.getChildren().size(); i++) {
                Shape[] newShape = block.getChildren().get(i).getValue().getOutputShapes(new Shape[]{currentShape});
                currentShape = newShape[0];
                System.out.println(block.getChildren().get(i).getKey() + " layer output : " + currentShape);
            }

            //
            int batchSize = 256;
            int numEpochs = Integer.getInteger("MAX_EPOCH", 10);

            double[] epochCount = new double[numEpochs];

            for (int i = 0; i < epochCount.length; i++) {
                epochCount[i] = (i + 1);
            }

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
            trainingChapter6(trainIter, testIter, numEpochs, trainer);
        }
    }

    public static void trainingChapter6(ArrayDataset trainIter,
                                        ArrayDataset testIter,
                                        int numEpochs, Trainer trainer) throws IOException, TranslateException {

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

        double[] trainLoss = evaluatorMetrics.get("train_epoch_SoftmaxCrossEntropyLoss");
        double[] trainAccuracy = evaluatorMetrics.get("train_epoch_Accuracy");
        double[] testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy");

        System.out.printf("loss %.3f,", trainLoss[numEpochs - 1]);
        System.out.printf(" train acc %.3f,", trainAccuracy[numEpochs - 1]);
        System.out.printf(" test acc %.3f\n", testAccuracy[numEpochs - 1]);
        System.out.printf("%.1f examples/sec \n", trainIter.size() / (avgTrainTimePerEpoch / Math.pow(10, 9)));
    }
}
