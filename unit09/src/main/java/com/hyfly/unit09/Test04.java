package com.hyfly.unit09;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import com.hyfly.utils.Functions;
import com.hyfly.utils.PlotUtils;
import com.hyfly.utils.attention.MultiHeadAttention;
import com.hyfly.utils.attention.PositionalEncoding;

public class Test04 {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            int numHiddens = 100;
            int numHeads = 5;
            MultiHeadAttention attention = new MultiHeadAttention(numHiddens, numHeads, 0.5f, false);

            //
            int batchSize = 2;
            int numQueries = 4;
            NDArray validLens = manager.create(new float[] {3, 2});
            NDArray X = manager.ones(new Shape(batchSize, numQueries, numHiddens));
            ParameterStore ps = new ParameterStore(manager, false);
            NDList input = new NDList(X, X, X, validLens);
            attention.initialize(manager, DataType.FLOAT32, input.getShapes());
            NDList result = attention.forward(ps, input, false);
            result.get(0).getShape();

            //
            int encodingDim = 32;
            int numSteps = 60;
            PositionalEncoding posEncoding = new PositionalEncoding(encodingDim, 0, 1000, manager);
            input = new NDList(manager.zeros(new Shape(1, numSteps, encodingDim)));
            X = posEncoding.forward(ps, input, false).get(0);
            NDArray P = posEncoding.P.get(new NDIndex(":, :{}, :", X.getShape().get(1)));

            double[][] plotX = new double[4][];
            double[][] plotY = new double[4][];
            for (int i = 0; i < 4; i++) {
                if (i == 0) {
                    plotX[i] = manager.arange(numSteps).toType(DataType.FLOAT64, false).toDoubleArray();
                } else {
                    plotX[i] = plotX[i - 1];
                }
                plotY[i] =
                        Functions.floatToDoubleArray(
                                P.get(new NDIndex("0, :, {},", i + 6)).toFloatArray());
            }


            PlotUtils.plot(
                    plotX,
                    plotY,
                    new String[] {"Col6", "Col7", "Col8", "Col9"},
                    "Row (position)",
                    "");

            //
            for (int i = 0; i < 8; i++) {
                System.out.println(i + " in binary is " + Integer.toBinaryString(i));
            }

            //
            P = P.get(new NDIndex("0, :, :")).expandDims(0).expandDims(0);
            PlotUtils.showHeatmaps(
                    P, "Column (encoding dimension)", "Row (position)", new String[] {""}, 500, 700);
        }
    }
}
