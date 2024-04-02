package com.hyfly.unit03;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Activation;
import ai.djl.training.GradientCollector;
import lombok.extern.slf4j.Slf4j;
import tech.tablesaw.api.FloatColumn;
import tech.tablesaw.api.Table;


@Slf4j
public class Test01 {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray x = manager.arange(-8.0f, 8.0f, 0.1f);
            x.setRequiresGradient(true);
            NDArray y = Activation.relu(x);

            // Converting the data into float arrays to render them in a plot.
            float[] X = x.toFloatArray();
            float[] Y = y.toFloatArray();

            Table data = Table.create("Data").addColumns(
                    FloatColumn.create("X", X),
                    FloatColumn.create("relu(x)", Y)
            );
//            render(LinePlot.create("", data, "x", "relu(X)"), "text/html");

            try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
                y = Activation.relu(x);
                collector.backward(y);
            }

            NDArray res = x.getGradient();
            float[] X1 = x.toFloatArray();
            float[] Y1 = res.toFloatArray();

            Table data1 = Table.create("Data").addColumns(
                    FloatColumn.create("X", X1),
                    FloatColumn.create("grad of relu", Y1)
            );
//            render(LinePlot.create("", data1, "x", "grad of relu"), "text/html");


            NDArray y2 = Activation.sigmoid(x);
            float[] Y2 = y.toFloatArray();

            Table data2 = Table.create("Data").addColumns(
                    FloatColumn.create("X", X),
                    FloatColumn.create("sigmoid(x)", Y2)
            );
//            render(LinePlot.create("", data2, "x", "sigmoid(X)"));
            try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
                y = Activation.sigmoid(x);
                collector.backward(y);
            }

            NDArray res3 = x.getGradient();
            float[] X3 = x.toFloatArray();
            float[] Y3 = res.toFloatArray();

            Table data3 = Table.create("Data").addColumns(
                    FloatColumn.create("X", X3),
                    FloatColumn.create("grad of sigmoid", Y3)
            );
//            render(LinePlot.create("", data3, "x", "grad of sigmoid"), "text/html");
        }
    }
}
