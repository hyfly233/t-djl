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
import com.hyfly.utils.TrainingChapter11;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class Test10 {

    public static void main(String[] args) throws Exception {
        AirfoilRandomAccess airfoil = TrainingChapter11.getDataCh11(10, 1500);

        TrainingChapter11.LossTime lossTime = trainAdam(0.01f, 1, 2, airfoil);

        //
        Tracker lrt = Tracker.fixed(0.01f);
        Optimizer adam = Optimizer.adam().optLearningRateTracker(lrt).build();

        TrainingChapter11.trainConciseCh11(adam, airfoil, 2);
    }

    public static NDList initAdamStates(int featureDimension) {
        NDManager manager = NDManager.newBaseManager();
        NDArray vW = manager.zeros(new Shape(featureDimension, 1));
        NDArray vB = manager.zeros(new Shape(1));
        NDArray sW = manager.zeros(new Shape(featureDimension, 1));
        NDArray sB = manager.zeros(new Shape(1));
        return new NDList(vW, sW, vB, sB);
    }

    public static TrainingChapter11.LossTime trainAdam(float lr, float time, int numEpochs, AirfoilRandomAccess airfoil) throws IOException, TranslateException {
        int featureDimension = airfoil.getColumnNames().size();
        Map<String, Float> hyperparams = new HashMap<>();
        hyperparams.put("lr", lr);
        hyperparams.put("time", time);
        return TrainingChapter11.trainCh11(Optimization::adam,
                initAdamStates(featureDimension),
                hyperparams, airfoil,
                featureDimension, numEpochs);
    }
}
