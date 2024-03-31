package com.hyfly.unit11;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import com.hyfly.unit11.entity.Residual;
import lombok.extern.slf4j.Slf4j;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@Slf4j
public class Test02 {

    public static void main(String[] args) throws Exception {
        int numClass = 10;
        // This model uses a smaller convolution kernel, stride, and padding and
        // removes the maximum pooling layer
        SequentialBlock net = new SequentialBlock();
        net.add(Conv2d.builder()
                        .setFilters(64)
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .build())
                .add(BatchNorm.builder().build())
                .add(Activation::relu)
                .add(resnetBlock(64, 2, true))
                .add(resnetBlock(128, 2, false))
                .add(resnetBlock(256, 2, false))
                .add(resnetBlock(512, 2, false))
                .add(Pool.globalAvgPool2dBlock())
                .add(Linear.builder().setUnits(numClass).build());

        //
        Model model = Model.newInstance("training-multiple-gpus-1");
        model.setBlock(net);

        Loss loss = Loss.softmaxCrossEntropyLoss();

        Tracker lrt = Tracker.fixed(0.1f);
        Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

        DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                .optOptimizer(sgd) // Optimizer (loss function)
                .optInitializer(new NormalInitializer(0.01f), Parameter.Type.WEIGHT) // setting the initializer
                .optDevices(Engine.getInstance().getDevices(1)) // setting the number of GPUs needed
                .addEvaluator(new Accuracy()) // Model Accuracy
                .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

        Trainer trainer = model.newTrainer(config);

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray X = manager.randomUniform(0f, 1.0f, new Shape(4, 1, 28, 28));
            trainer.initialize(X.getShape());

            NDList[] res = Batchifier.STACK.split(new NDList(X), 4, true);

            ParameterStore parameterStore = new ParameterStore(manager, true);

            System.out.println(net.forward(parameterStore, new NDList(res[0]), false).singletonOrThrow());
            System.out.println(net.forward(parameterStore, new NDList(res[1]), false).singletonOrThrow());
            System.out.println(net.forward(parameterStore, new NDList(res[2]), false).singletonOrThrow());
            System.out.println(net.forward(parameterStore, new NDList(res[3]), false).singletonOrThrow());

            NDArray ndArray = net.getChildren().values().get(0).getParameters().get("weight").getArray().get(new NDIndex("0:1"));
            log.info("{}", ndArray.toDebugString(true));

            int numEpochs = Integer.getInteger("MAX_EPOCH", 10);

            double[] testAccuracy = new double[0];
            double[] epochCount;

            epochCount = new double[numEpochs];

            for (int i = 0; i < epochCount.length; i++) {
                epochCount[i] = (i + 1);
            }

            Map<String, double[]> evaluatorMetrics = new HashMap<>();
            double avgTrainTimePerEpoch = 0;

            Table data = null;
            // We will check if we have at least 1 GPU available. If yes, we run the training on 1 GPU.
            if (Engine.getInstance().getGpuCount() >= 1) {
                train(numEpochs, trainer, 256, testAccuracy);

                data = Table.create("Data");
                data = data.addColumns(
                        DoubleColumn.create("X", epochCount),
                        DoubleColumn.create("testAccuracy", testAccuracy)
                );
            }

            data = Table.create("Data");

            // We will check if we have more than 1 GPU available. If yes, we run the training on 2 GPU.
            if (Engine.getInstance().getGpuCount() > 1) {

                X = manager.randomUniform(0f, 1.0f, new Shape(1, 1, 28, 28));

                model = Model.newInstance("training-multiple-gpus-2");
                model.setBlock(net);

                loss = Loss.softmaxCrossEntropyLoss();

                lrt = Tracker.fixed(0.2f);
                sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

                config = new DefaultTrainingConfig(loss)
                        .optOptimizer(sgd) // Optimizer (loss function)
                        .optInitializer(new NormalInitializer(0.01f), Parameter.Type.WEIGHT) // setting the initializer
                        .optDevices(Engine.getInstance().getDevices(2)) // setting the number of GPUs needed
                        .addEvaluator(new Accuracy()) // Model Accuracy
                        .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

                trainer = model.newTrainer(config);

                trainer.initialize(X.getShape());

                evaluatorMetrics = new HashMap<>();
                avgTrainTimePerEpoch = 0;

                train(numEpochs, trainer, 512, testAccuracy);

                data = data.addColumns(
                        DoubleColumn.create("X", epochCount),
                        DoubleColumn.create("testAccuracy", testAccuracy)
                );
            }
        }
    }

    public static SequentialBlock resnetBlock(int numChannels, int numResiduals, boolean isFirstBlock) {

        SequentialBlock blk = new SequentialBlock();
        for (int i = 0; i < numResiduals; i++) {

            if (i == 0 && !isFirstBlock) {
                blk.add(new Residual(numChannels, true, new Shape(2, 2)));
            } else {
                blk.add(new Residual(numChannels, false, new Shape(1, 1)));
            }
        }
        return blk;
    }

    public static void train(int numEpochs, Trainer trainer, int batchSize, double[] testAccuracy) throws IOException, TranslateException {

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

        Map<String, double[]> evaluatorMetrics = new HashMap<>();
        double avgTrainTime = 0;

        trainer.setMetrics(new Metrics());

        EasyTrain.fit(trainer, numEpochs, trainIter, testIter);

        Metrics metrics = trainer.getMetrics();

        trainer.getEvaluators().stream()
                .forEach(evaluator -> {
                    evaluatorMetrics.put("validate_epoch_" + evaluator.getName(), metrics.getMetric("validate_epoch_" + evaluator.getName()).stream()
                            .mapToDouble(x -> x.getValue().doubleValue()).toArray());
                });

        avgTrainTime = metrics.mean("epoch");
        testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy");
        System.out.printf("test acc %.2f\n", testAccuracy[numEpochs - 1]);
        System.out.println(avgTrainTime / Math.pow(10, 9) + " sec/epoch \n");
    }
}
