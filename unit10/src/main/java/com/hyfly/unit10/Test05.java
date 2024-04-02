package com.hyfly.unit10;

import ai.djl.Model;
import ai.djl.basicdataset.tabular.AirfoilRandomAccess;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import com.hyfly.unit10.entity.LossTime;
import com.hyfly.unit10.entity.Optimization;
import com.hyfly.unit10.entity.TrainerConsumer;
import com.hyfly.utils.Accumulator;
import com.hyfly.utils.Functions;
import com.hyfly.utils.StopWatch;
import com.hyfly.utils.Training;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.api.LinePlot;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

@Slf4j
public class Test05 {

    public static void main(String[] args) throws IOException, TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            StopWatch stopWatch = new StopWatch();
            NDArray A = manager.zeros(new Shape(256, 256));
            NDArray B = manager.randomNormal(new Shape(256, 256));
            NDArray C = manager.randomNormal(new Shape(256, 256));

            // 逐元素计算A=BC
            stopWatch.start();
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    A.set(new NDIndex(i, j),
                            B.get(new NDIndex(String.format("%d, :", i)))
                                    .dot(C.get(new NDIndex(String.format(":, %d", j)))));
                }
            }
            stopWatch.stop();

            // 逐列计算A=BC
            stopWatch.start();
            for (int j = 0; j < 256; j++) {
                A.set(new NDIndex(String.format(":, %d", j)), B.dot(C.get(new NDIndex(String.format(":, %d", j)))));
            }
            stopWatch.stop();

            // 一次性计算A=BC
            stopWatch.start();
            A = B.dot(C);
            stopWatch.stop();

            // Multiply and add count as separate operations (fused in practice)
            float[] gigaflops = new float[stopWatch.getTimes().size()];
            for (int i = 0; i < stopWatch.getTimes().size(); i++) {
                gigaflops[i] = (float) (2 / stopWatch.getTimes().get(i));
            }
            System.out.printf("Performance in Gigaflops: element %.3f, column %.3f, full %.3f%n", gigaflops[0], gigaflops[1], gigaflops[2]);

            stopWatch.start();
            for (int j = 0; j < 256; j += 64) {
                A.set(new NDIndex(String.format(":, %d:%d", j, j + 64)),
                        B.dot(C.get(new NDIndex(String.format(":, %d:%d", j, j + 64)))));
            }
            stopWatch.stop();

            System.out.printf("Performance in Gigaflops: block %.3f\n%n", 2 / stopWatch.getTimes().get(3));

            //
            LossTime gdRes = trainSgd(1f, 1500, 10);

            //
            LossTime sgdRes = trainSgd(0.005f, 1, 2);

            //
            LossTime mini1Res = trainSgd(0.4f, 100, 2);

            //
            LossTime mini2Res = trainSgd(0.05f, 10, 2);

            //
            float[] time = ArrayUtils.addAll(ArrayUtils.addAll(gdRes.time, sgdRes.time),
                    ArrayUtils.addAll(mini1Res.time, mini2Res.time));
            float[] loss = ArrayUtils.addAll(ArrayUtils.addAll(gdRes.loss, sgdRes.loss),
                    ArrayUtils.addAll(mini1Res.loss, mini2Res.loss));
            String[] type = ArrayUtils.addAll(ArrayUtils.addAll(getTypeArray(gdRes, "gd"),
                            getTypeArray(sgdRes, "sgd")),
                    ArrayUtils.addAll(getTypeArray(mini1Res, "batch size = 100"),
                            getTypeArray(mini1Res, "batch size = 10")));
            Table data = Table.create("data")
                    .addColumns(
                            DoubleColumn.create("log time (sec)", Functions.floatToDoubleArray(convertLogScale(time))),
                            DoubleColumn.create("loss", Functions.floatToDoubleArray(loss)),
                            StringColumn.create("type", type)
                    );
            LinePlot.create("loss vs. time", data, "log time (sec)", "loss", "type");

            //
            AirfoilRandomAccess airfoilDataset = getDataCh11(10, 1500);

            Tracker lrt = Tracker.fixed(0.05f);
            Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

            trainConciseCh11(sgd, airfoilDataset, 2);
        }
    }

    public static AirfoilRandomAccess getDataCh11(int batchSize, int n) throws IOException, TranslateException {
        // Load data
        AirfoilRandomAccess airfoil = AirfoilRandomAccess.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, true)
                .optNormalize(true)
                .optLimit(n)
                .build();
        return airfoil;
    }

    public static float evaluateLoss(Iterable<Batch> dataIterator, NDArray w, NDArray b) {
        Accumulator metric = new Accumulator(2);  // sumLoss, numExamples

        for (Batch batch : dataIterator) {
            NDArray X = batch.getData().head();
            NDArray y = batch.getLabels().head();
            NDArray yHat = Training.linreg(X, w, b);
            float lossSum = Training.squaredLoss(yHat, y).sum().getFloat();

            metric.add(new float[]{lossSum, (float) y.size()});
            batch.close();
        }
        return metric.get(0) / metric.get(1);
    }

    public static void plotLossEpoch(float[] loss, float[] epoch) {
        Table data = Table.create("data")
                .addColumns(
                        DoubleColumn.create("epoch", Functions.floatToDoubleArray(epoch)),
                        DoubleColumn.create("loss", Functions.floatToDoubleArray(loss))
                );
//        display(LinePlot.create("loss vs. epoch", data, "epoch", "loss"));
    }

    public static float[] arrayListToFloat(ArrayList<Double> arrayList) {
        float[] ret = new float[arrayList.size()];

        for (int i = 0; i < arrayList.size(); i++) {
            ret[i] = arrayList.get(i).floatValue();
        }
        return ret;
    }

    public static LossTime trainCh11(TrainerConsumer trainer, NDList states, Map<String, Float> hyperparams,
                                     AirfoilRandomAccess dataset,
                                     int featureDim, int numEpochs) throws IOException, TranslateException {
        NDManager manager = NDManager.newBaseManager();
        NDArray w = manager.randomNormal(0, 0.01f, new Shape(featureDim, 1), DataType.FLOAT32);
        NDArray b = manager.zeros(new Shape(1));

        w.setRequiresGradient(true);
        b.setRequiresGradient(true);

        NDList params = new NDList(w, b);
        int n = 0;
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        float lastLoss = -1;
        ArrayList<Double> loss = new ArrayList<>();
        ArrayList<Double> epoch = new ArrayList<>();

        for (int i = 0; i < numEpochs; i++) {
            for (Batch batch : dataset.getData(manager)) {
                int len = (int) dataset.size() / batch.getSize();  // number of batches
                NDArray X = batch.getData().head();
                NDArray y = batch.getLabels().head();

                NDArray l;
                try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                    NDArray yHat = Training.linreg(X, params.get(0), params.get(1));
                    l = Training.squaredLoss(yHat, y).mean();
                    gc.backward(l);
                }

                trainer.train(params, states, hyperparams);
                n += X.getShape().get(0);

                if (n % 200 == 0) {
                    stopWatch.stop();
                    lastLoss = evaluateLoss(dataset.getData(manager), params.get(0), params.get(1));
                    loss.add((double) lastLoss);
                    double lastEpoch = 1.0 * n / X.getShape().get(0) / len;
                    epoch.add(lastEpoch);
                    stopWatch.start();
                }

                batch.close();
            }
        }
        float[] lossArray = arrayListToFloat(loss);
        float[] epochArray = arrayListToFloat(epoch);
        plotLossEpoch(lossArray, epochArray);
        System.out.printf("loss: %.3f, %.3f sec/epoch\n", lastLoss, stopWatch.avg());
        float[] timeArray = arrayListToFloat(stopWatch.cumsum());
        return new LossTime(lossArray, timeArray);
    }

    public static LossTime trainSgd(float lr, int batchSize, int numEpochs) throws IOException, TranslateException {
        AirfoilRandomAccess dataset = getDataCh11(batchSize, 1500);
        int featureDim = dataset.getColumnNames().size();

        Map<String, Float> hyperparams = new HashMap<>();
        hyperparams.put("lr", lr);

        return trainCh11(Optimization::sgd, new NDList(), hyperparams, dataset, featureDim, numEpochs);
    }

    public static String[] getTypeArray(LossTime lossTime, String name) {
        String[] type = new String[lossTime.time.length];
        for (int i = 0; i < type.length; i++) {
            type[i] = name;
        }
        return type;
    }

    // Converts a float array to a log scale
    public static float[] convertLogScale(float[] array) {
        float[] newArray = new float[array.length];
        for (int i = 0; i < array.length; i++) {
            newArray[i] = (float) Math.log10(array[i]);
        }
        return newArray;
    }

    public static void trainConciseCh11(Optimizer sgd, AirfoilRandomAccess dataset,
                                        int numEpochs) throws IOException, TranslateException {
        // Initialization
        try (NDManager manager = NDManager.newBaseManager()) {


            SequentialBlock net = new SequentialBlock();
            Linear linear = Linear.builder().setUnits(1).build();
            net.add(linear);
            net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);

            Model model = Model.newInstance("concise implementation");
            model.setBlock(net);

            Loss loss = Loss.l2Loss();

            DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                    .optOptimizer(sgd)
                    .addEvaluator(new Accuracy()) // Model Accuracy
                    .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

            Trainer trainer = model.newTrainer(config);

            int n = 0;
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();

            trainer.initialize(new Shape(10, 5));

            Metrics metrics = new Metrics();
            trainer.setMetrics(metrics);

            float lastLoss = -1;

            ArrayList<Double> lossArray = new ArrayList<>();
            ArrayList<Double> epochArray = new ArrayList<>();

            for (Batch batch : trainer.iterateDataset(dataset)) {
                int len = (int) dataset.size() / batch.getSize();  // number of batches

                NDArray X = batch.getData().head();
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();

                n += X.getShape().get(0);

                if (n % 200 == 0) {
                    stopWatch.stop();
                    stopWatch.stop();
                    lastLoss = evaluateLoss(dataset.getData(manager), linear.getParameters().get(0).getValue().getArray()
                                    .reshape(new Shape(dataset.getColumnNames().size(), 1)),
                            linear.getParameters().get(1).getValue().getArray());

                    lossArray.add((double) lastLoss);
                    double lastEpoch = 1.0 * n / X.getShape().get(0) / len;
                    epochArray.add(lastEpoch);
                    stopWatch.start();
                }
                batch.close();
            }
            plotLossEpoch(arrayListToFloat(lossArray), arrayListToFloat(epochArray));

            System.out.printf("loss: %.3f, %.3f sec/epoch\n", lastLoss, stopWatch.avg());
        }
    }
}
