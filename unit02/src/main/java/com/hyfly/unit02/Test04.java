package com.hyfly.unit02;

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
import com.hyfly.utils.DataPoints;
import com.hyfly.utils.Training;
import lombok.extern.slf4j.Slf4j;
import tech.tablesaw.api.FloatColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.api.ScatterPlot;

@Slf4j
public class Test04 {

    public static void main(String[] args) throws Exception {

        try (NDManager manager = NDManager.newBaseManager()) {
            // 3.2.1. 生成数据集
            NDArray trueW = manager.create(new float[]{2, -3.4f});
            float trueB = 4.2f;

            DataPoints dp = DataPoints.syntheticData(manager, trueW, trueB, 1000);
            NDArray features = dp.getX();
            NDArray labels = dp.getY();

            log.info("features: [{}, {}]\n", features.get(0).getFloat(0), features.get(0).getFloat(1));
            log.info("label: {}", labels.getFloat(0));

            float[] X = features.get(new NDIndex(":, 1")).toFloatArray();
            float[] y = labels.toFloatArray();

            Table data = Table.create("Data")
                    .addColumns(
                            FloatColumn.create("X", X),
                            FloatColumn.create("y", y)
                    );

            ScatterPlot.create("Synthetic Data", data, "X", "y");

            log.info("------------------------");

            // 3.2.2. 读取数据集
            int batchSize = 10;

            ArrayDataset dataset = new ArrayDataset.Builder()
                    .setData(features) // Set the Features
                    .optLabels(labels) // Set the Labels
                    .setSampling(batchSize, false) // set the batch size and random sampling to false
                    .build();

            for (Batch batch : dataset.getData(manager)) {
                // Call head() to get the first NDArray
                NDArray x1 = batch.getData().head();
                NDArray y1 = batch.getLabels().head();
                log.info("x1: {}", x1.toDebugString(true));
                log.info("y1: {}", y1.toDebugString(true));
                // Don't forget to close the batch!
                batch.close();
                break;
            }

            // 3.2.3. 初始化模型参数
            NDArray w = manager.randomNormal(0, 0.01f, new Shape(2, 1), DataType.FLOAT32);
            NDArray b = manager.zeros(new Shape(1));
            NDList params = new NDList(w, b);

            // 3.2.7. 训练
            float lr = 0.03f;  // 学习率
            int numEpochs = 3;  // 迭代次数

            // 为 NDArray 的梯度分配内存
            for (NDArray param : params) {
                param.setRequiresGradient(true);
            }

            for (int epoch = 0; epoch < numEpochs; epoch++) {
                // Assuming the number of examples can be divided by the batch size, all
                // the examples in the training dataset are used once in one epoch
                // iteration. The features and tags of minibatch examples are given by ndArrayX
                // and ndArrayY respectively.
                for (Batch batch : dataset.getData(manager)) {
                    NDArray ndArrayX = batch.getData().head();
                    NDArray ndArrayY = batch.getLabels().head();

                    try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                        // Minibatch loss in ndArrayX and ndArrayY
                        NDArray l = Training.squaredLoss(Training.linreg(ndArrayX, params.get(0), params.get(1)), ndArrayY);
                        gc.backward(l);  // Compute gradient on l with respect to w and b
                    }
                    Training.sgd(params, lr, batchSize);  // Update parameters using their gradient

                    batch.close();
                }
                NDArray trainL = Training.squaredLoss(Training.linreg(features, params.get(0), params.get(1)), labels);
                log.info("epoch {}, loss {}\n", epoch + 1, trainL.mean().getFloat());
            }

            float[] wf = trueW.sub(params.get(0).reshape(trueW.getShape())).toFloatArray();
            log.info("Error in estimating w: [{}, {}]", wf[0], wf[1]);
            log.info("Error in estimating b: {}", trueB - params.get(1).getFloat());
        }
    }
}
