package com.hyfly.unit10;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.hyfly.utils.GradDescUtils;
import lombok.extern.slf4j.Slf4j;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

@Slf4j
public class Test04 {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            float eta = 0.01f;
            Supplier<Float> lr = () -> 1f; // Constant Learning Rate

            BiFunction<Float, Float, Float> f = (x1, x2) -> x1 * x1 + 2 * x2 * x2; // Objective

            BiFunction<Float, Float, Float[]> gradf = (x1, x2) -> new Float[]{2 * x1, 4 * x2}; // Gradient

            Supplier<Float> finalLr = lr;
            Function<Float[], Float[]> sgd = (state) -> {
                Float x1 = state[0];
                Float x2 = state[1];
                Float s1 = state[2];
                Float s2 = state[3];

                Float[] g = gradf.apply(x1, x2);
                Float g1 = g[0];
                Float g2 = g[1];

                g1 += getRandomNormal(0f, 0.1f, manager);
                g2 += getRandomNormal(0f, 0.1f, manager);
                Float etaT = eta * finalLr.get();
                return new Float[]{x1 - etaT * g1, x2 - etaT * g2, 0f, 0f};
            };

            GradDescUtils.showTrace2d(f, GradDescUtils.train2d(sgd, 50));

            //
            AtomicInteger ctr = new AtomicInteger(1);

            Supplier<Float> exponential = () -> {
                ctr.addAndGet(1);
                return (float) Math.exp(-0.1 * ctr.get());
            };

            lr = exponential; // Set up learning rate
            GradDescUtils.showTrace2d(f, GradDescUtils.train2d(sgd, 1000));

            AtomicInteger ctr1 = new AtomicInteger(1);

            Supplier<Float> polynomial = () -> {
                ctr1.addAndGet(1);
                return (float) Math.pow(1 + 0.1 * ctr1.get(), -0.5);
            };

            lr = polynomial; // Set up learning rate
            GradDescUtils.showTrace2d(f, GradDescUtils.train2d(sgd, 1000));
        }
    }

    // Sample once from a normal distribution
    public static float getRandomNormal(float mean, float sd, NDManager manager) {
        return manager.randomNormal(mean, sd, new Shape(1), DataType.FLOAT32).getFloat();
    }
}
