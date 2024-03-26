package com.hyfly.unit01;

import lombok.extern.slf4j.Slf4j;

import java.util.function.Function;

@Slf4j
public class Test04 {

    public static void main(String[] args) {
        Function<Double, Double> f = x -> (3 * Math.pow(x, 2) - 4 * x);

        double h = 0.1;
        for (int i = 0; i < 5; i++) {
            log.info("h=" + String.format("%.5f", h) +
                    ", numerical limit=" +
                    String.format("%.5f", numericalLim(f, 1, h)));
            h *= 0.1;
        }
    }

    public static Double numericalLim(Function<Double, Double> f, double x, double h) {
        return (f.apply(x + h) - f.apply(x)) / h;
    }
}
