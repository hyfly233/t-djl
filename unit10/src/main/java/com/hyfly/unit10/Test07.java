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
import com.hyfly.utils.GradDescUtils;
import com.hyfly.utils.TrainingChapter11;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

@Slf4j
public class Test07 {

    public static void main(String[] args) throws Exception {
        float eta = 0.4f;

        float finalEta = eta;
        Function<Float[], Float[]> adagrad2d = (state) -> {
            Float x1 = state[0], x2 = state[1], s1 = state[2], s2 = state[3];
            float eps = (float) 1e-6;
            float g1 = 0.2f * x1;
            float g2 = 4 * x2;
            s1 += g1 * g1;
            s2 += g2 * g2;
            x1 -= finalEta / (float) Math.sqrt(s1 + eps) * g1;
            x2 -= finalEta / (float) Math.sqrt(s2 + eps) * g2;
            return new Float[]{x1, x2, s1, s2};
        };

        BiFunction<Float, Float, Float> f2d = (x1, x2) -> 0.1f * x1 * x1 + 2 * x2 * x2;

        GradDescUtils.showTrace2d(f2d, GradDescUtils.train2d(adagrad2d, 20));

        //
        eta = 2;
        GradDescUtils.showTrace2d(f2d, GradDescUtils.train2d(adagrad2d, 20));

        //
        AirfoilRandomAccess airfoil = TrainingChapter11.getDataCh11(10, 1500);
        TrainingChapter11.LossTime lossTime = trainAdagrad(0.1f, 2, airfoil);

        //
        Tracker lrt = Tracker.fixed(0.1f);
        Optimizer adagrad = Optimizer.adagrad().optLearningRateTracker(lrt).build();

        TrainingChapter11.trainConciseCh11(adagrad, airfoil, 2);
    }

    public static NDList initAdagradStates(int featureDimension) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray sW = manager.zeros(new Shape(featureDimension, 1));
            NDArray sB = manager.zeros(new Shape(1));
            return new NDList(sW, sB);
        }
    }

    public static TrainingChapter11.LossTime trainAdagrad(float lr, int numEpochs, AirfoilRandomAccess airfoil) throws IOException, TranslateException {
        int featureDimension = airfoil.getColumnNames().size();
        Map<String, Float> hyperparams = new HashMap<>();
        hyperparams.put("lr", lr);
        return TrainingChapter11.trainCh11(Optimization::adagrad,
                initAdagradStates(featureDimension),
                hyperparams, airfoil, featureDimension, numEpochs);
    }
}
