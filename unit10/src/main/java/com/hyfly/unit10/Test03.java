package com.hyfly.unit10;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.hyfly.unit10.entity.Weights;
import com.hyfly.utils.Functions;
import lombok.extern.slf4j.Slf4j;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.ScatterTrace;

import java.util.ArrayList;
import java.util.function.Function;

@Slf4j
public class Test03 {

    public static void main(String[] args) {
        Function<Float, Float> f = x -> x * x; // Objective Function
        Function<Float, Float> gradf = x1 -> 2 * x1; // Its Derivative

        try (NDManager manager = NDManager.newBaseManager()) {
            float[] res = gd(0.2f, gradf);

            showTrace(res);

            showTrace(gd(0.05f, gradf));

            showTrace(gd(1.1f, gradf));

            float c = (float) (0.15f * Math.PI);

            Function<Float, Float> f1 = x -> x * (float) Math.cos(c * x);

            Function<Float, Float> gradf1 = x -> (float) (Math.cos(c * x) - c * x * Math.sin(c * x));

            showTrace(gd(2, gradf1));
        }
    }

    public static float[] gd(float eta, Function<Float, Float> gradf) {
        float x = 10f;
        float[] results = new float[11];
        results[0] = x;

        for (int i = 0; i < 10; i++) {
            x -= eta * gradf.apply(x);
            results[i + 1] = x;
        }
        System.out.printf("epoch 10, x: %f\n", x);
        return results;
    }

    /* Saved in GradDescUtils.java */
    public static void plotGD(float[] x, float[] y, float[] segment, Function<Float, Float> func,
                              int width, int height) {
        // Function Line
        ScatterTrace trace = ScatterTrace.builder(Functions.floatToDoubleArray(x),
                        Functions.floatToDoubleArray(y))
                .mode(ScatterTrace.Mode.LINE)
                .build();

        // GD Line
        ScatterTrace trace2 = ScatterTrace.builder(Functions.floatToDoubleArray(segment),
                        Functions.floatToDoubleArray(Functions.callFunc(segment, func)))
                .mode(ScatterTrace.Mode.LINE)
                .build();

        // GD Points
        ScatterTrace trace3 = ScatterTrace.builder(Functions.floatToDoubleArray(segment),
                        Functions.floatToDoubleArray(Functions.callFunc(segment, func)))
                .build();

        Layout layout = Layout.builder()
                .height(height)
                .width(width)
                .showLegend(false)
                .build();

//        display(new Figure(layout, trace, trace2, trace3));
    }

    /* Saved in GradDescUtils.java */
    public static void showTrace(float[] res, NDManager manager, Function<Float, Float> f) {
        float n = 0;
        for (int i = 0; i < res.length; i++) {
            if (Math.abs(res[i]) > n) {
                n = Math.abs(res[i]);
            }
        }
        NDArray fLineND = manager.arange(-n, n, 0.01f);
        float[] fLine = fLineND.toFloatArray();
        plotGD(fLine, Functions.callFunc(fLine, f), res, f, 500, 400);
    }

    /* Saved in GradDescUtils.java */
    /* Optimize a 2D objective function with a customized trainer. */
    public static ArrayList<Weights> train2d(Function<Float[], Float[]> trainer, int steps) {
        // s1和s2是稍后将使用的内部状态变量
        float x1 = -5f, x2 = -2f, s1 = 0f, s2 = 0f;
        ArrayList<Weights> results = new ArrayList<>();
        results.add(new Weights(x1, x2));
        for (int i = 1; i < steps + 1; i++) {
            Float[] step = trainer.apply(new Float[]{x1, x2, s1, s2});
            x1 = step[0];
            x2 = step[1];
            s1 = step[2];
            s2 = step[3];
            results.add(new Weights(x1, x2));
            System.out.printf("epoch %d, x1 %f, x2 %f\n", i, x1, x2);
        }
        return results;
    }
}
