package com.hyfly.unit10;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.hyfly.utils.Functions;
import lombok.extern.slf4j.Slf4j;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.ScatterTrace;

import java.util.function.Function;

@Slf4j
public class Test02 {

    public static void main(String[] args) {
        Function<Float, Float> f = x -> 0.5f * x * x; // 凸函数
        Function<Float, Float> g = x -> (float) Math.cos(Math.PI * x); // 非凸函数
        Function<Float, Float> h = x -> (float) Math.exp(0.5f * x); // 凸函数

        NDManager manager = NDManager.newBaseManager();

        NDArray X = manager.arange(-2f, 2f, 0.01f);
        float[] x = X.toFloatArray();
        float[] segment = new float[]{-1.5f, 1f};

        float[] fx = Functions.callFunc(x, f);
        float[] gx = Functions.callFunc(x, g);
        float[] hx = Functions.callFunc(x, h);

//        display(plotLineAndSegment(x, fx, segment, f, 350, 300));
//        display(plotLineAndSegment(x, gx, segment, g, 350, 300));
//        display(plotLineAndSegment(x, hx, segment, h, 350, 300));

        Function<Float, Float> f1 = x1 -> (x1 - 1) * (x1 - 1) * (x1 + 1);

        float[] fx1 = Functions.callFunc(x, f1);
        plotLineAndSegment(x, fx1, segment, f, 400, 350);
    }

    // ScatterTrace.builder() does not support float[],
    // so we must convert to a double array first
    public static double[] floatToDoubleArray(float[] x) {
        double[] ret = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            ret[i] = x[i];
        }
        return ret;
    }

    public static Figure plotLineAndSegment(float[] x, float[] y, float[] segment, Function<Float, Float> func,
                                            int width, int height) {
        ScatterTrace trace = ScatterTrace.builder(floatToDoubleArray(x), floatToDoubleArray(y))
                .mode(ScatterTrace.Mode.LINE)
                .build();

        ScatterTrace trace2 = ScatterTrace.builder(floatToDoubleArray(segment),
                        new double[]{func.apply(segment[0]),
                                func.apply(segment[1])})
                .mode(ScatterTrace.Mode.LINE)
                .build();

        Layout layout = Layout.builder()
                .height(height)
                .width(width)
                .showLegend(false)
                .build();

        return new Figure(layout, trace, trace2);
    }
}
