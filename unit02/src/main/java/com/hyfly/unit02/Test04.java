package com.hyfly.unit02;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import com.hyfly.unit02.entity.DataPoints;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Test04 {

    public static void main(String[] args) throws Exception {

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray trueW = manager.create(new float[]{2, -3.4f});
            float trueB = 4.2f;

            DataPoints dp = DataPoints.syntheticData(manager, trueW, trueB, 1000);
            NDArray features = dp.getX();
            NDArray labels = dp.getY();

            // 3.2.2. 读取数据集
            int batchSize = 10;

            ArrayDataset dataset = new ArrayDataset.Builder()
                    .setData(features) // Set the Features
                    .optLabels(labels) // Set the Labels
                    .setSampling(batchSize, false) // set the batch size and random sampling to false
                    .build();

            //
            NDArray w = manager.randomNormal(0, 0.01f, new Shape(2, 1), DataType.FLOAT32);
            NDArray b = manager.zeros(new Shape(1));
            NDList params = new NDList(w, b);

            float lr = 0.03f;  // Learning Rate
            int numEpochs = 3;  // Number of Iterations

            // Attach Gradients
            for (NDArray param : params) {
                param.setRequiresGradient(true);
            }

            for (int epoch = 0; epoch < numEpochs; epoch++) {
                // Assuming the number of examples can be divided by the batch size, all
                // the examples in the training dataset are used once in one epoch
                // iteration. The features and tags of minibatch examples are given by X
                // and y respectively.
                for (Batch batch : dataset.getData(manager)) {
                    NDArray X = batch.getData().head();
                    NDArray y = batch.getLabels().head();

                    try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                        // Minibatch loss in X and y
                        NDArray l = squaredLoss(linreg(X, params.get(0), params.get(1)), y);
                        gc.backward(l);  // Compute gradient on l with respect to w and b
                    }
                    sgd(params, lr, batchSize);  // Update parameters using their gradient

                    batch.close();
                }
                NDArray trainL = squaredLoss(linreg(features, params.get(0), params.get(1)), labels);
                System.out.printf("epoch %d, loss %f\n", epoch + 1, trainL.mean().getFloat());

                float[] wf = trueW.sub(params.get(0).reshape(trueW.getShape())).toFloatArray();
                System.out.printf("Error in estimating w: [%f, %f]%n", wf[0], wf[1]);
                System.out.printf("Error in estimating b: %f%n", trueB - params.get(1).getFloat());
            }
        }
    }

    public static NDArray linreg(NDArray X, NDArray w, NDArray b) {
        return X.dot(w).add(b);
    }

    public static NDArray squaredLoss(NDArray yHat, NDArray y) {
        return (yHat.sub(y.reshape(yHat.getShape()))).mul
                ((yHat.sub(y.reshape(yHat.getShape())))).div(2);
    }

    public static void sgd(NDList params, float lr, int batchSize) {
        for (int i = 0; i < params.size(); i++) {
            NDArray param = params.get(i);
            // Update param
            // param = param - param.gradient * lr / batchSize
            param.subi(param.getGradient().mul(lr).div(batchSize));
        }
    }
}
