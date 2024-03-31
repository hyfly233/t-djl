package com.hyfly.unit10;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import tech.tablesaw.api.FloatColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.api.LinePlot;

import java.util.function.Function;

@Slf4j
public class Test01 {

    public static void main(String[] args) {
        Function<Float, Float> f = x -> x * (float) Math.cos(Math.PI * x);

        Function<Float, Float> g = x -> f.apply(x) + 0.2f * (float) Math.cos(5 * Math.PI * x);

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray X = manager.arange(0.5f, 1.5f, 0.01f);
            float[] x = X.toFloatArray();
            float[] fx = callFunc(x, f);
            float[] gx = callFunc(x, g);

            String[] grouping = new String[x.length * 2];
            for (int i = 0; i < x.length; i++) {
                grouping[i] = "Expected Risk";
                grouping[i + x.length] = "Empirical Risk";
            }

            Table data = Table.create("Data")
                    .addColumns(
                            FloatColumn.create("x", ArrayUtils.addAll(x, x)),
                            FloatColumn.create("risk", ArrayUtils.addAll(fx, gx)),
                            StringColumn.create("grouping", grouping)
                    );

            LinePlot.create("Risk", data, "x", "risk", "grouping");

            X = manager.arange(-1.0f, 2.0f, 0.01f);
            x = X.toFloatArray();
            fx = callFunc(x, f);

            data = Table.create("Data")
                    .addColumns(
                            FloatColumn.create("x", x),
                            FloatColumn.create("f(x)", fx)
                    );

            LinePlot.create("x * cos(pi * x)", data, "x", "f(x)");

            //
            Function<Float, Float> cube = x1 -> x1 * x1 * x1;

            X = manager.arange(-2.0f, 2.0f, 0.01f);
            x = X.toFloatArray();
            fx = callFunc(x, cube);

            data = Table.create("Data")
                    .addColumns(
                            FloatColumn.create("x", x),
                            FloatColumn.create("f(x)", fx)
                    );

            LinePlot.create("x^3", data, "x", "f(x)");

            Function<Float, Float> tanh = x2 -> (float) Math.tanh(x2);

            X = manager.arange(-2.0f, 5.0f, 0.01f);
            x = X.toFloatArray();
            fx = callFunc(x, tanh);

            data = Table.create("Data")
                    .addColumns(
                            FloatColumn.create("x", x),
                            FloatColumn.create("f(x)", fx)
                    );

            LinePlot.create("tanh", data, "x", "f(x)");
        }


    }

    public static float[] callFunc(float[] x, Function<Float, Float> func) {
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = func.apply(x[i]);
        }
        return y;
    }

}
