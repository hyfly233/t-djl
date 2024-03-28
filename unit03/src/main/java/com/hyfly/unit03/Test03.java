package com.hyfly.unit03;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

@Slf4j
public class Test03 {

    public static void main(String[] args) throws Exception {
        SequentialBlock net = new SequentialBlock();
        net.add(Blocks.batchFlattenBlock(784));
        net.add(Linear.builder().setUnits(256).build());
        net.add(Activation::relu);
        net.add(Linear.builder().setUnits(10).build());
        net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);

        int batchSize = 256;
        int numEpochs = Integer.getInteger("MAX_EPOCH", 10);
        double[] trainLoss;
        double[] testAccuracy;
        double[] epochCount;
        double[] trainAccuracy;

        trainLoss = new double[numEpochs];
        trainAccuracy = new double[numEpochs];
        testAccuracy = new double[numEpochs];
        epochCount = new double[numEpochs];

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

        for (int i = 0; i < epochCount.length; i++) {
            epochCount[i] = (i + 1);
        }

        Map<String, double[]> evaluatorMetrics = new HashMap<>();

        Tracker lrt = Tracker.fixed(0.5f);
        Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

        Loss loss = Loss.softmaxCrossEntropyLoss();

        DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                .optOptimizer(sgd) // Optimizer (loss function)
                .optDevices(Engine.getInstance().getDevices(1)) // single GPU
                .addEvaluator(new Accuracy()) // Model Accuracy
                .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

        try (Model model = Model.newInstance("mlp")) {
            model.setBlock(net);

            try (Trainer trainer = model.newTrainer(config)) {

                trainer.initialize(new Shape(1, 784));
                trainer.setMetrics(new Metrics());

                EasyTrain.fit(trainer, numEpochs, trainIter, testIter);
                // collect results from evaluators
                Metrics metrics = trainer.getMetrics();

                trainer.getEvaluators().stream()
                        .forEach(evaluator -> {
                            evaluatorMetrics.put("train_epoch_" + evaluator.getName(), metrics.getMetric("train_epoch_" + evaluator.getName()).stream()
                                    .mapToDouble(x -> x.getValue().doubleValue()).toArray());
                            evaluatorMetrics.put("validate_epoch_" + evaluator.getName(), metrics.getMetric("validate_epoch_" + evaluator.getName()).stream()
                                    .mapToDouble(x -> x.getValue().doubleValue()).toArray());
                        });
            }
        }

        trainLoss = evaluatorMetrics.get("train_epoch_SoftmaxCrossEntropyLoss");
        trainAccuracy = evaluatorMetrics.get("train_epoch_Accuracy");
        testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy");

        String[] lossLabel = new String[trainLoss.length + testAccuracy.length + trainAccuracy.length];

        Arrays.fill(lossLabel, 0, trainLoss.length, "test acc");
        Arrays.fill(lossLabel, trainAccuracy.length, trainLoss.length + trainAccuracy.length, "train acc");
        Arrays.fill(lossLabel, trainLoss.length + trainAccuracy.length,
                trainLoss.length + testAccuracy.length + trainAccuracy.length, "train loss");

        Table data = Table.create("Data").addColumns(
                DoubleColumn.create("epochCount", ArrayUtils.addAll(epochCount, ArrayUtils.addAll(epochCount, epochCount))),
                DoubleColumn.create("loss", ArrayUtils.addAll(testAccuracy, ArrayUtils.addAll(trainAccuracy, trainLoss))),
                StringColumn.create("lossLabel", lossLabel)
        );

//        render(LinearPlot.create("", data, "epochCount", "loss", "lossLabel"),"text/html");
    }
}
