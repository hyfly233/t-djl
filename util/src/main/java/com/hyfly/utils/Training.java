package com.hyfly.utils;

import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.util.Map;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

public class Training {

    /**
     * 3.2.4. 定义模型
     * Linear regression 模型: 将输入和权重相乘，然后加上偏差
     *
     * @param X input 输入特征
     * @param w weight 模型权重
     * @param b bias 模型偏差
     * @return X * w + b
     */
    public static NDArray linreg(NDArray X, NDArray w, NDArray b) {
        return X.matMul(w).add(b);
    }

    /**
     * 3.2.5. 定义损失函数
     * 平方损失函数 最小二乘损失
     *
     * @param yHat 预测值
     * @param y    真实值
     * @return 损失值
     */
    public static NDArray squaredLoss(NDArray yHat, NDArray y) {
        return (yHat.sub(y.reshape(yHat.getShape())))
                .mul((yHat.sub(y.reshape(yHat.getShape()))))
                .div(2);
    }

    /**
     * 3.2.6. 定义优化算法
     * 随机梯度下降
     *
     * @param params    模型参数
     * @param lr        学习率
     * @param batchSize 批量大小
     */
    public static void sgd(NDList params, float lr, int batchSize) {
        for (NDArray param : params) {
            // Update param in place.
            // param = param - param.gradient * lr / batchSize
            param.subi(param.getGradient().mul(lr).div(batchSize));
        }
    }

    /**
     * Allows to do gradient calculations on a subManager. This is very useful when you are training
     * on a lot of epochs. This subManager could later be closed and all NDArrays generated from the
     * calculations in this function will be cleared from memory when subManager is closed. This is
     * always a great practice but the impact is most notable when there is lot of data on various
     * epochs.
     */
    public static void sgd(NDList params, float lr, int batchSize, NDManager subManager) {
        for (NDArray param : params) {
            // Update param in place.
            // param = param - param.gradient * lr / batchSize
            NDArray gradient = param.getGradient();
            gradient.attach(subManager);
            param.subi(gradient.mul(lr).div(batchSize));
        }
    }

    public static float accuracy(NDArray yHat, NDArray y) {
        // Check size of 1st dimension greater than 1
        // to see if we have multiple samples
        if (yHat.getShape().size(1) > 1) {
            // Argmax gets index of maximum args for given axis 1
            // Convert yHat to same dataType as y (int32)
            // Sum up number of true entries
            return yHat.argMax(1)
                    .toType(DataType.INT32, false)
                    .eq(y.toType(DataType.INT32, false))
                    .sum()
                    .toType(DataType.FLOAT32, false)
                    .getFloat();
        }
        return yHat.toType(DataType.INT32, false)
                .eq(y.toType(DataType.INT32, false))
                .sum()
                .toType(DataType.FLOAT32, false)
                .getFloat();
    }

    public static double trainingChapter6(
            ArrayDataset trainIter,
            ArrayDataset testIter,
            int numEpochs,
            Trainer trainer,
            Map<String, double[]> evaluatorMetrics)
            throws IOException, TranslateException {

        trainer.setMetrics(new Metrics());

        EasyTrain.fit(trainer, numEpochs, trainIter, testIter);

        Metrics metrics = trainer.getMetrics();

        trainer.getEvaluators()
                .forEach(
                        evaluator -> {
                            evaluatorMetrics.put(
                                    "train_epoch_" + evaluator.getName(),
                                    metrics.getMetric("train_epoch_" + evaluator.getName()).stream()
                                            .mapToDouble(x -> x.getValue().doubleValue())
                                            .toArray());
                            evaluatorMetrics.put(
                                    "validate_epoch_" + evaluator.getName(),
                                    metrics
                                            .getMetric("validate_epoch_" + evaluator.getName())
                                            .stream()
                                            .mapToDouble(x -> x.getValue().doubleValue())
                                            .toArray());
                        });

        return metrics.mean("epoch");
    }

    /* Softmax-regression-scratch */
    public static float evaluateAccuracy(UnaryOperator<NDArray> net, Iterable<Batch> dataIterator) {
        Accumulator metric = new Accumulator(2); // numCorrectedExamples, numExamples
        for (Batch batch : dataIterator) {
            NDArray X = batch.getData().head();
            NDArray y = batch.getLabels().head();
            metric.add(new float[]{accuracy(net.apply(X), y), (float) y.size()});
            batch.close();
        }
        return metric.get(0) / metric.get(1);
    }
    /* End Softmax-regression-scratch */

    public static float evaluateAccuracy(NDList params, int numInputs, Iterable<Batch> dataIterator) {
        Accumulator metric = new Accumulator(2); // numCorrectedExamples, numExamples
        for (Batch batch : dataIterator) {
            NDArray X = batch.getData().head();
            NDArray y = batch.getLabels().head();

            NDArray currentW = params.get(0);
            NDArray currentB = params.get(1);
            NDArray softmax = Softmax.softmax(X.reshape(new Shape(-1, numInputs)).dot(currentW).add(currentB));

            metric.add(new float[]{accuracy(softmax, y), (float) y.size()});
            batch.close();
        }
        return metric.get(0) / metric.get(1);
    }


    /* MLP */
    /* Evaluate the loss of a model on the given dataset */
    public static float evaluateLoss(
            UnaryOperator<NDArray> net,
            Iterable<Batch> dataIterator,
            BinaryOperator<NDArray> loss) {
        Accumulator metric = new Accumulator(2); // sumLoss, numExamples

        for (Batch batch : dataIterator) {
            NDArray X = batch.getData().head();
            NDArray y = batch.getLabels().head();
            metric.add(
                    new float[]{loss.apply(net.apply(X), y).sum().getFloat(), (float) y.size()});
            batch.close();
        }
        return metric.get(0) / metric.get(1);
    }
    /* End MLP */

    /**
     * ArrayDataset 数据集
     *
     * @param features  特征
     * @param labels    标签
     * @param batchSize 批量大小
     * @param shuffle   是否打乱
     * @return ArrayDataset
     */
    public static ArrayDataset loadArray(NDArray features, NDArray labels, int batchSize, boolean shuffle) {
        return new ArrayDataset.Builder()
                .setData(features) // set the features
                .optLabels(labels) // set the labels
                .setSampling(batchSize, shuffle) // set the batch size and random sampling
                .build();
    }
}
