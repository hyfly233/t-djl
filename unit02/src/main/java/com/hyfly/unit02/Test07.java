package com.hyfly.unit02;

import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import com.hyfly.utils.FashionMnistUtils;
import com.hyfly.utils.Softmax;
import lombok.extern.slf4j.Slf4j;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.function.UnaryOperator;

import static com.hyfly.utils.Training.accuracy;

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
            NDArray yHat = manager.create(new float[][]{{0.1f, 0.3f, 0.6f}, {0.3f, 0.2f, 0.5f}});
            NDArray ndArray = yHat.get(new NDIndex(":, {}", manager.create(new int[]{0, 2})));
            log.info(ndArray.toDebugString(true));

            // 3.6.5. 计算分类准确率
            NDArray y = manager.create(new int[]{0, 2});
            float v = accuracy(yHat, y) / y.size();
            log.info("accuracy: {}", v);

//            evaluateAccuracy(Net.net(params, X, numInputs), validationSet.getData(manager));
            // 3.6.6. 训练模型

            int numEpochs = 5;
            float lr = 0.1f;

//            trainCh3(Net::net, trainingSet, validationSet, LossFunction::crossEntropy, numEpochs, Updater::updater);
//
//            predictCh3(Net::net, validationSet, 6, manager);
        }

    }

    @FunctionalInterface
    public static interface ParamConsumer {
        void accept(NDList params, float lr, int batchSize);
    }

//    public static float[] trainEpochCh3(UnaryOperator<NDArray> net, Iterable<Batch> trainIter, BinaryOperator<NDArray> loss, ParamConsumer updater) {
//        Accumulator metric = new Accumulator(3); // trainLossSum, trainAccSum, numExamples
//
//        // Attach Gradients
//        for (NDArray param : params) {
//            param.setRequiresGradient(true);
//        }
//
//        for (Batch batch : trainIter) {
//            NDArray X = batch.getData().head();
//            NDArray y = batch.getLabels().head();
//            X = X.reshape(new Shape(-1, numInputs));
//
//            try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
//                // Minibatch loss in X and y
//                NDArray yHat = net.apply(X);
//                NDArray l = loss.apply(yHat, y);
//                gc.backward(l);  // Compute gradient on l with respect to w and b
//                metric.add(new float[]{l.sum().toType(DataType.FLOAT32, false).getFloat(),
//                        accuracy(yHat, y),
//                        (float) y.size()});
//                gc.close();
//            }
//            updater.accept(params, lr, batch.getSize());  // Update parameters using their gradient
//
//            batch.close();
//        }
//        // Return trainLoss, trainAccuracy
//        return new float[]{metric.get(0) / metric.get(2), metric.get(1) / metric.get(2)};
//    }
//
//    public static void trainCh3(UnaryOperator<NDArray> net, Dataset trainDataset, Dataset testDataset,
//                                BinaryOperator<NDArray> loss, int numEpochs, ParamConsumer updater)
//            throws IOException, TranslateException {
//        Animator animator = new Animator();
//        for (int i = 1; i <= numEpochs; i++) {
//            float[] trainMetrics = trainEpochCh3(net, trainDataset.getData(manager), loss, updater);
//            float accuracy = evaluateAccuracy(net, testDataset.getData(manager));
//            float trainAccuracy = trainMetrics[1];
//            float trainLoss = trainMetrics[0];
//
//            animator.add(i, accuracy, trainAccuracy, trainLoss);
//            System.out.printf("Epoch %d: Test Accuracy: %f\n", i, accuracy);
//            System.out.printf("Train Accuracy: %f\n", trainAccuracy);
//            System.out.printf("Train Loss: %f\n", trainLoss);
//        }
//    }

    // Number should be < batchSize for this function to work properly
    public static BufferedImage predictCh3(UnaryOperator<NDArray> net, ArrayDataset dataset, int number, NDManager manager)
            throws IOException, TranslateException {
        final int SCALE = 4;
        final int WIDTH = 28;
        final int HEIGHT = 28;

        int[] predLabels = new int[number];

        for (Batch batch : dataset.getData(manager)) {
            NDArray X = batch.getData().head();
            int[] yHat = net.apply(X).argMax(1).toType(DataType.INT32, false).toIntArray();
            for (int i = 0; i < number; i++) {
                predLabels[i] = yHat[i];
            }
            break;
        }

        return FashionMnistUtils.showImages(dataset, predLabels, WIDTH, HEIGHT, SCALE, manager);
    }
}
