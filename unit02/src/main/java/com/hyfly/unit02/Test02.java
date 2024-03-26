package com.hyfly.unit02;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import tech.tablesaw.api.FloatColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.api.LinePlot;
import tech.tablesaw.plotly.components.Figure;

import java.util.Arrays;

@Slf4j
public class Test02 {

    public static void main(String[] args) {
        int start = -7;
        int end = 14;
        float step = 0.01f;
        int count = (int) (end / step);

        float[] x = new float[count];

        for (int i = 0; i < count; i++) {
            x[i] = start + i * step;
        }

        float[] y1 = normal(x, 0, 1);
        float[] y2 = normal(x, 0, 2);
        float[] y3 = normal(x, 3, 1);

        String[] params = new String[x.length * 3];

        Arrays.fill(params, 0, x.length, "mean 0, var 1");
        Arrays.fill(params, x.length, x.length * 2, "mean 0, var 2");
        Arrays.fill(params, x.length * 2, x.length * 3, "mean 3, var 1");

        Table normalDistributions = Table.create("normal")
                .addColumns(
                        FloatColumn.create("z", combine3(x, x, x)),
                        FloatColumn.create("p(z)", combine3(y1, y2, y3)),
                        StringColumn.create("params", params)
                );

        Figure figure = LinePlot.create("Normal Distributions", normalDistributions, "z", "p(z)", "params");

        log.info(figure.toString());
    }

    public static float[] normal(float[] z, float mu, float sigma) {
        float[] dist = new float[z.length];
        for (int i = 0; i < z.length; i++) {
            float p = 1.0f / (float) Math.sqrt(2 * Math.PI * sigma * sigma);
            dist[i] = p * (float) Math.pow(Math.E, -0.5 / (sigma * sigma) * (z[i] - mu) * (z[i] - mu));
        }
        return dist;
    }

    public static float[] combine3(float[] x, float[] y, float[] z) {
        return ArrayUtils.addAll(ArrayUtils.addAll(x, y), z);
    }
}
