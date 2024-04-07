package com.hyfly.unit02;

import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import com.hyfly.utils.*;
import lombok.extern.slf4j.Slf4j;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.function.BinaryOperator;


/**
 * 3.6. Softmax 回归的从零开始实现
 */
@Slf4j
public class Test07 {

    public static void main(String[] args) throws Exception {
        int batchSize = 256;
        boolean randomShuffle = true;

        // get training and validation dataset
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

        // 3.6.1. 初始化模型参数
        int numInputs = 784;
        int numOutputs = 10;

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray W = manager.randomNormal(0, 0.01f, new Shape(numInputs, numOutputs), DataType.FLOAT32);
            NDArray b = manager.zeros(new Shape(numOutputs), DataType.FLOAT32);
            NDList params = new NDList(W, b);

            // 3.6.2. 定义 softmax 操作
            log.info("3.6.2. 定义 softmax 操作 -----------------");
            NDArray X = manager.create(new int[][]{{1, 2, 3}, {4, 5, 6}});
            log.info(X.sum(new int[]{0}, true).toDebugString(true));
            log.info(X.sum(new int[]{1}, true).toDebugString(true));
            log.info(X.sum(new int[]{0, 1}, true).toDebugString(true));

            //
            X = manager.randomNormal(new Shape(2, 5));
            NDArray xprob = Softmax.softmax(X);
            log.info(xprob.toDebugString(true));
            log.info(xprob.sum(new int[]{1}).toDebugString(true));

            // 3.6.4. 定义损失函数
            log.info("3.6.4. 定义损失函数 -----------------");
            NDArray yHat = manager.create(new float[][]{{0.1f, 0.3f, 0.6f}, {0.3f, 0.2f, 0.5f}});
            NDArray ndArray = yHat.get(new NDIndex(":, {}", manager.create(new int[]{0, 2})));
            log.info(ndArray.toDebugString(true));

            // 3.6.5. 计算分类准确率
            log.info("3.6.5. 计算分类准确率 -----------------");
            NDArray y = manager.create(new int[]{0, 2});
            float v = Training.accuracy(yHat, y) / y.size();
            log.info("Training.accuracy: {}", v);

            float v1 = Training.evaluateAccuracy(params, numInputs, validationSet.getData(manager));
            log.info("Training.evaluateAccuracy: {}", v1);

            // 3.6.6. 训练模型
            log.info("3.6.6. 训练模型 -----------------");
            int numEpochs = 5;
            float lr = 0.1f;

            trainCh3(trainingSet, validationSet, LossFunction::crossEntropy, numEpochs, Updater::updater, params, numInputs, lr, manager);

            BufferedImage image = predictCh3(params, numInputs, validationSet, 6, manager);
            log.info("image: {}", image);
        }

    }

    @FunctionalInterface
    public interface ParamConsumer {
        void accept(NDList params, float lr, int batchSize);
    }

    public static float[] trainEpochCh3(Iterable<Batch> trainIter,
                                        BinaryOperator<NDArray> loss,
                                        ParamConsumer updater,
                                        NDList params,
                                        int numInputs,
                                        float lr) {
        Accumulator metric = new Accumulator(3); // trainLossSum, trainAccSum, numExamples

        // Attach Gradients
        for (NDArray param : params) {
            param.setRequiresGradient(true);
        }

        for (Batch batch : trainIter) {
            NDArray X = batch.getData().head();
            NDArray y = batch.getLabels().head();
            X = X.reshape(new Shape(-1, numInputs));

            try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                // Minibatch loss in X and y
                NDArray currentW = params.get(0);
                NDArray currentB = params.get(1);

                NDArray yHat = Softmax.softmax(X.reshape(new Shape(-1, numInputs)).dot(currentW).add(currentB));
                NDArray l = loss.apply(yHat, y);
                gc.backward(l);  // Compute gradient on l with respect to w and b
                metric.add(new float[]{l.sum().toType(DataType.FLOAT32, false).getFloat(),
                        Training.accuracy(yHat, y),
                        (float) y.size()});
                gc.close();
            }
            updater.accept(params, lr, batch.getSize());  // Update parameters using their gradient

            batch.close();
        }
        // Return trainLoss, trainAccuracy
        return new float[]{metric.get(0) / metric.get(2), metric.get(1) / metric.get(2)};
    }

    public static void trainCh3(Dataset trainDataset,
                                Dataset testDataset,
                                BinaryOperator<NDArray> loss,
                                int numEpochs,
                                ParamConsumer updater,
                                NDList params,
                                int numInputs,
                                float lr,
                                NDManager manager)
            throws IOException, TranslateException {
        Animator animator = new Animator();
        for (int i = 1; i <= numEpochs; i++) {
            float[] trainMetrics = trainEpochCh3(trainDataset.getData(manager), loss, updater, params, numInputs, lr);
            float accuracy = Training.evaluateAccuracy(params, numInputs, testDataset.getData(manager));
            float trainAccuracy = trainMetrics[1];
            float trainLoss = trainMetrics[0];

            animator.add(i, accuracy, trainAccuracy, trainLoss);
            log.info("Epoch {}: Test Accuracy: {}", i, accuracy);
            log.info("Train Accuracy: {}", trainAccuracy);
            log.info("Train Loss: {}", trainLoss);
        }
    }

    public static BufferedImage predictCh3(NDList params, int numInputs, ArrayDataset dataset, int number, NDManager manager)
            throws IOException, TranslateException {
        final int SCALE = 4;
        final int WIDTH = 28;
        final int HEIGHT = 28;

        int[] predLabels = new int[number];

        for (Batch batch : dataset.getData(manager)) {
            NDList data = batch.getData();
            NDArray X = data.head();
            NDArray currentW = params.get(0);
            NDArray currentB = params.get(1);
            NDArray softmax = Softmax.softmax(X.reshape(new Shape(-1, numInputs)).dot(currentW).add(currentB));

            int[] yHat = softmax.argMax(1).toType(DataType.INT32, false).toIntArray();
            System.arraycopy(yHat, 0, predLabels, 0, number);
            break;
        }

        return FashionMnistUtils.showImages(dataset, predLabels, WIDTH, HEIGHT, SCALE, manager);
    }
}
