package com.hyfly.unit10;

import ai.djl.basicdataset.tabular.AirfoilRandomAccess;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import com.hyfly.unit10.entity.Optimization;
import com.hyfly.utils.Functions;
import com.hyfly.utils.GradDescUtils;
import com.hyfly.utils.TrainingChapter11;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.api.LinePlot;
import tech.tablesaw.plotly.components.Axis;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.ScatterTrace;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

@Slf4j
public class Test06 {

    public static void main(String[] args) throws Exception {
        float eta = 0.4f;
        BiFunction<Float, Float, Float> f2d = (x1, x2) -> 0.1f * x1 * x1 + 2 * x2 * x2;

        float finalEta1 = eta;
        Function<Float[], Float[]> gd2d = (state) -> {
            Float x1 = state[0], x2 = state[1], s1 = state[2], s2 = state[3];
            return new Float[]{x1 - finalEta1 * 0.2f * x1, x2 - finalEta1 * 4 * x2, 0f, 0f};
        };

        GradDescUtils.showTrace2d(f2d, GradDescUtils.train2d(gd2d, 20));

        //
        eta = 0.6f;
        GradDescUtils.showTrace2d(f2d, GradDescUtils.train2d(gd2d, 20));

        //
        eta = 0.6f;
        float beta = 0.5f;

        float finalEta = eta;
        float finalBeta = beta;
        Function<Float[], Float[]> momentum2d = (state) -> {
            Float x1 = state[0], x2 = state[1], v1 = state[2], v2 = state[3];
            v1 = finalBeta * v1 + 0.2f * x1;
            v2 = finalBeta * v2 + 4 * x2;
            return new Float[]{x1 - finalEta * v1, x2 - finalEta * v2, v1, v2};
        };

        GradDescUtils.showTrace2d(f2d, GradDescUtils.train2d(momentum2d, 20));

        //
        eta = 0.6f;
        beta = 0.25f;
        GradDescUtils.showTrace2d(f2d, GradDescUtils.train2d(momentum2d, 20));

        //
        try (NDManager manager = NDManager.newBaseManager()) {
            float[] gammas = new float[]{0.95f, 0.9f, 0.6f, 0f};

            NDArray timesND = manager.arange(40f);
            float[] times = timesND.toFloatArray();

            plotGammas(times, gammas, 600, 400);

            //
            AirfoilRandomAccess airfoil = TrainingChapter11.getDataCh11(10, 1500);


            trainMomentum(0.02f, 0.5f, 2, airfoil);

            //
            trainMomentum(0.01f, 0.9f, 2, airfoil);

            //
            trainMomentum(0.005f, 0.9f, 2, airfoil);

            //
            Tracker lrt = Tracker.fixed(0.005f);
            Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).optMomentum(0.9f).build();

            TrainingChapter11.trainConciseCh11(sgd, airfoil, 2);

            //
            float[] lambdas = new float[]{0.1f, 1f, 10f, 19f};
            eta = 0.1f;

            float[] time = new float[0];
            float[] convergence = new float[0];
            String[] lambda = new String[0];
            for (float lam : lambdas) {
                float[] timeTemp = new float[20];
                float[] convergenceTemp = new float[20];
                String[] lambdaTemp = new String[20];
                for (int i = 0; i < timeTemp.length; i++) {
                    timeTemp[i] = i;
                    convergenceTemp[i] = (float) Math.pow(1 - eta * lam, i);
                    lambdaTemp[i] = String.format("lambda = %.2f", lam);
                }
                time = ArrayUtils.addAll(time, timeTemp);
                convergence = ArrayUtils.addAll(convergence, convergenceTemp);
                lambda = ArrayUtils.addAll(lambda, lambdaTemp);
            }

            Table data = Table.create("data")
                    .addColumns(
                            DoubleColumn.create("time", Functions.floatToDoubleArray(time)),
                            DoubleColumn.create("convergence", Functions.floatToDoubleArray(convergence)),
                            StringColumn.create("lambda", lambda)
                    );

            LinePlot.create("convergence vs. time", data, "time", "convergence", "lambda");
        }
    }

    /* Saved in GradDescUtils.java */
    public static Figure plotGammas(float[] time, float[] gammas,
                                    int width, int height) {
        double[] gamma1 = new double[time.length];
        double[] gamma2 = new double[time.length];
        double[] gamma3 = new double[time.length];
        double[] gamma4 = new double[time.length];

        // Calculate all gammas over time
        for (int i = 0; i < time.length; i++) {
            gamma1[i] = Math.pow(gammas[0], i);
            gamma2[i] = Math.pow(gammas[1], i);
            gamma3[i] = Math.pow(gammas[2], i);
            gamma4[i] = Math.pow(gammas[3], i);
        }

        // Gamma 1 Line
        ScatterTrace gamma1trace = ScatterTrace.builder(Functions.floatToDoubleArray(time),
                        gamma1)
                .mode(ScatterTrace.Mode.LINE)
                .name(String.format("gamma = %.2f", gammas[0]))
                .build();

        // Gamma 2 Line
        ScatterTrace gamma2trace = ScatterTrace.builder(Functions.floatToDoubleArray(time),
                        gamma2)
                .mode(ScatterTrace.Mode.LINE)
                .name(String.format("gamma = %.2f", gammas[1]))
                .build();

        // Gamma 3 Line
        ScatterTrace gamma3trace = ScatterTrace.builder(Functions.floatToDoubleArray(time),
                        gamma3)
                .mode(ScatterTrace.Mode.LINE)
                .name(String.format("gamma = %.2f", gammas[2]))
                .build();

        // Gamma 4 Line
        ScatterTrace gamma4trace = ScatterTrace.builder(Functions.floatToDoubleArray(time),
                        gamma4)
                .mode(ScatterTrace.Mode.LINE)
                .name(String.format("gamma = %.2f", gammas[3]))
                .build();

        Axis xAxis = Axis.builder()
                .title("time")
                .build();

        Layout layout = Layout.builder()
                .height(height)
                .width(width)
                .xAxis(xAxis)
                .build();

        return new Figure(layout, gamma1trace, gamma2trace, gamma3trace, gamma4trace);
    }

    public static NDList initMomentumStates(int featureDim) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray vW = manager.zeros(new Shape(featureDim, 1));
            NDArray vB = manager.zeros(new Shape(1));
            return new NDList(vW, vB);
        }
    }

    public static TrainingChapter11.LossTime trainMomentum(float lr, float momentum, int numEpochs, AirfoilRandomAccess airfoil)
            throws IOException, TranslateException {
        int featureDim = airfoil.getColumnNames().size();
        Map<String, Float> hyperparams = new HashMap<>();
        hyperparams.put("lr", lr);
        hyperparams.put("momentum", momentum);
        return TrainingChapter11.trainCh11(Optimization::sgdMomentum, initMomentumStates(featureDim), hyperparams, airfoil, featureDim, numEpochs);
    }
}
