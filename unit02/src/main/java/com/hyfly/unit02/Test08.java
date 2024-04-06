package com.hyfly.unit02;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Test08 {

    public static void main(String[] args) throws Exception {
        int batchSize = 256;
        boolean randomShuffle = true;

        // Get Training and Validation Datasets
        FashionMnist trainingSet = FashionMnist.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, randomShuffle)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();


        FashionMnist validationSet = FashionMnist.builder()
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, false)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();

        // 3.7.1. 初始化模型参数
        try (NDManager manager = NDManager.newBaseManager();
             Model model = Model.newInstance("softmax-regression")) {

            SequentialBlock net = new SequentialBlock();
            net.add(Blocks.batchFlattenBlock(28 * 28)); // flatten input
            net.add(Linear.builder().setUnits(10).build()); // set 10 output channels

            model.setBlock(net);

            // 3.7.2. 重新审视 Softmax 的实现
            Loss loss = Loss.softmaxCrossEntropyLoss();

            // 3.7.3. 优化算法
            Tracker lrt = Tracker.fixed(0.1f);
            Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

            // 3.7.4. Trainer 的初始化配置
            DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                    .optOptimizer(sgd) // Optimizer
                    .optDevices(manager.getEngine().getDevices(1)) // single GPU
                    .addEvaluator(new Accuracy()) // Model Accuracy
                    .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

            Trainer trainer = model.newTrainer(config);

            // 3.7.5. 初始化模型参数
            trainer.initialize(new Shape(1, 28 * 28)); // Input Images are 28 x 28

            // 3.7.6. 运行性能指标
            Metrics metrics = new Metrics();
            trainer.setMetrics(metrics);

            // 3.7.7. 训练
            int numEpochs = 3;

            EasyTrain.fit(trainer, numEpochs, trainingSet, validationSet);
            var result = trainer.getTrainingResult();
            log.info("Result: {}", result);
        }
    }
}
