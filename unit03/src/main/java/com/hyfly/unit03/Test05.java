package com.hyfly.unit03;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import com.hyfly.unit03.entity.InitParams;
import com.hyfly.utils.DataPoints;
import com.hyfly.utils.Training;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;

import java.io.IOException;
import java.util.Arrays;


@Slf4j
public class Test05 {

    public static void main(String[] args) throws Exception {
        int nTrain = 20;
        int nTest = 100;
        int numInputs = 200;
        int batchSize = 5;

        float trueB = 0.05f;
        NDManager manager = NDManager.newBaseManager();
        NDArray trueW = manager.ones(new Shape(numInputs, 1));
        trueW = trueW.mul(0.01);

        DataPoints trainData = DataPoints.syntheticData(manager, trueW, trueB, nTrain);

        ArrayDataset trainIter = Training.loadArray(trainData.getX(), trainData.getY(), batchSize, true);

        DataPoints testData = DataPoints.syntheticData(manager, trueW, trueB, nTest);

        ArrayDataset testIter = Training.loadArray(testData.getX(), testData.getY(), batchSize, false);

        Loss l2loss = Loss.l2Loss();

        double[] trainLoss = new double[0];
        double[] testLoss = new double[0];
        double[] epochCount = new double[0];

        train(0f, trainIter, manager, batchSize, testData, trainData);

        String[] lossLabel = new String[trainLoss.length + testLoss.length];

        Arrays.fill(lossLabel, 0, testLoss.length, "test");
        Arrays.fill(lossLabel, testLoss.length, trainLoss.length + testLoss.length, "train");

        Table data = Table.create("Data").addColumns(
                DoubleColumn.create("epochCount", ArrayUtils.addAll(epochCount, epochCount)),
                DoubleColumn.create("loss", ArrayUtils.addAll(testLoss, trainLoss)),
                StringColumn.create("lossLabel", lossLabel)
        );

//        render(LinePlot.create("", data, "epochCount", "loss", "lossLabel"),"text/html");


        // calling training with weight decay lambda = 3.0
        train(3f, trainIter, manager, batchSize, testData, trainData);

        String[] lossLabel1 = new String[trainLoss.length + testLoss.length];

        Arrays.fill(lossLabel1, 0, testLoss.length, "test");
        Arrays.fill(lossLabel1, testLoss.length, trainLoss.length + testLoss.length, "train");

        Table data1 = Table.create("Data").addColumns(
                DoubleColumn.create("epochCount", ArrayUtils.addAll(epochCount, epochCount)),
                DoubleColumn.create("loss", ArrayUtils.addAll(testLoss, trainLoss)),
                StringColumn.create("lossLabel", lossLabel1)
        );

//        render(LinePlot.create("", data1, "epochCount", "loss", "lossLabel"),"text/html");
    }

    public static NDArray l2Penalty(NDArray w) {
        return ((w.pow(2)).sum()).div(2);
    }

    public static void train(float lambd, ArrayDataset trainIter, NDManager manager, int batchSize, DataPoints testData, DataPoints trainData) throws IOException, TranslateException {

        InitParams initParams = new InitParams();

        NDList params = new NDList(initParams.getW(), initParams.getB());

        int numEpochs = Integer.getInteger("MAX_EPOCH", 100);
        float lr = 0.003f;

        double[] trainLoss = new double[(numEpochs / 5)];
        double[] testLoss = new double[(numEpochs / 5)];
        double[] epochCount = new double[(numEpochs / 5)];

        for (int epoch = 1; epoch <= numEpochs; epoch++) {

            for (Batch batch : trainIter.getData(manager)) {

                NDArray X = batch.getData().head();
                NDArray y = batch.getLabels().head();

                NDArray w = params.get(0);
                NDArray b = params.get(1);

                try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                    // The L2 norm penalty term has been added, and broadcasting
                    // makes `l2Penalty(w)` a vector whose length is `batch_size`
                    NDArray l = Training.squaredLoss(Training.linreg(X, w, b), y).add(l2Penalty(w).mul(lambd));
                    gc.backward(l);  // Compute gradient on l with respect to w and b

                }

                batch.close();
                Training.sgd(params, lr, batchSize);  // Update parameters using their gradient
            }

            if (epoch % 5 == 0) {
                NDArray testL = Training.squaredLoss(Training.linreg(testData.getX(), params.get(0), params.get(1)), testData.getY());
                NDArray trainL = Training.squaredLoss(Training.linreg(trainData.getX(), params.get(0), params.get(1)), trainData.getY());

                epochCount[epoch / 5 - 1] = epoch;
                trainLoss[epoch / 5 - 1] = trainL.mean().log10().getFloat();
                testLoss[epoch / 5 - 1] = testL.mean().log10().getFloat();
            }

        }

        System.out.println("l1 norm of w: " + params.get(0).abs().sum());
    }

//    public void train_djl(float wd) throws IOException, TranslateException {
//
//        InitParams initParams = new InitParams();
//
//        NDList params = new NDList(initParams.getW(), initParams.getB());
//
//        int numEpochs = Integer.getInteger("MAX_EPOCH", 100);
//        float lr = 0.003f;
//
//        trainLoss = new double[(numEpochs/5)];
//        testLoss = new double[(numEpochs/5)];
//        epochCount = new double[(numEpochs/5)];
//
//        Tracker lrt = Tracker.fixed(lr);
//        Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();
//
//        Model model = Model.newInstance("mlp");
//
//        DefaultTrainingConfig config = new DefaultTrainingConfig(l2loss)
//                .optOptimizer(sgd) // Optimizer (loss function)
//                .optDevices(model.getNDManager().getEngine().getDevices(1)) // single CPU/GPU
//                .addEvaluator(new Accuracy()) // Model Accuracy
//                .addEvaluator(l2loss)
//                .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging
//
//        SequentialBlock net = new SequentialBlock();
//        Linear linearBlock = Linear.builder().optBias(true).setUnits(1).build();
//        net.add(linearBlock);
//
//        model.setBlock(net);
//        Trainer trainer = model.newTrainer(config);
//
//        trainer.initialize(new Shape(batchSize, 2));
//        for(int epoch = 1; epoch <= numEpochs; epoch++){
//
//            for(Batch batch : trainer.iterateDataset(trainIter)){
//
//                NDArray X = batch.getData().head();
//                NDArray y = batch.getLabels().head();
//
//                NDArray w = params.get(0);
//                NDArray b = params.get(1);
//
//                try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
//                    // Minibatch loss in X and y
//                    NDArray l = Training.squaredLoss(Training.linreg(X, w, b), y).add(l2Penalty(w).mul(wd));
//                    gc.backward(l);  // Compute gradient on l with respect to w and b
//
//                }
//                batch.close();
//                Training.sgd(params, lr, batchSize);  // Update parameters using their gradient
//            }
//
//            if(epoch % 5 == 0){
//                NDArray testL = Training.squaredLoss(Training.linreg(testData.getX(), params.get(0), params.get(1)), testData.getY());
//                NDArray trainL = Training.squaredLoss(Training.linreg(trainData.getX(), params.get(0), params.get(1)), trainData.getY());
//
//                epochCount[epoch/5 - 1] = epoch;
//                trainLoss[epoch/5 -1] = trainL.mean().log10().getFloat();
//                testLoss[epoch/5 -1] = testL.mean().log10().getFloat();
//            }
//
//        }
//        System.out.println("l1 norm of w: " + params.get(0).abs().sum());
//    }
}
