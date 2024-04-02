package com.hyfly.unit10;

import ai.djl.basicdataset.tabular.AirfoilRandomAccess;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.translate.TranslateException;
import com.hyfly.unit10.entity.Optimization;
import com.hyfly.utils.TrainingChapter11;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class Test09 {

    public static void main(String[] args) throws Exception {
        AirfoilRandomAccess airfoil = TrainingChapter11.getDataCh11(10, 1500);

        TrainingChapter11.LossTime lossTime = trainAdadelta(0.9f, 2, airfoil);

        //
        Optimizer adadelta = Optimizer.adadelta().optRho(0.9f).build();

        TrainingChapter11.trainConciseCh11(adadelta, airfoil, 2);
    }

    public static NDList initAdadeltaStates(int featureDimension) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray sW = manager.zeros(new Shape(featureDimension, 1));
            NDArray sB = manager.zeros(new Shape(1));
            NDArray deltaW = manager.zeros(new Shape(featureDimension, 1));
            NDArray deltaB = manager.zeros(new Shape(1));
            return new NDList(sW, deltaW, sB, deltaB);
        }
    }

    public static TrainingChapter11.LossTime trainAdadelta(float rho, int numEpochs, AirfoilRandomAccess airfoil) throws IOException, TranslateException {
        int featureDimension = airfoil.getColumnNames().size();
        Map<String, Float> hyperparams = new HashMap<>();
        hyperparams.put("rho", rho);
        return TrainingChapter11.trainCh11(Optimization::adadelta,
                initAdadeltaStates(featureDimension),
                hyperparams, airfoil,
                featureDimension, numEpochs);
    }
}
