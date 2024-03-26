package com.hyfly.unit02;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import cn.hutool.core.date.StopWatch;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Test01 {

    public static void main(String[] args) {
        int n = 10000;

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray a = manager.ones(new Shape(n));
            NDArray b = manager.ones(new Shape(n));

            NDArray c = manager.zeros(new Shape(n));

            StopWatch stopWatch = new StopWatch();
            stopWatch.start();

            for (int i = 0; i < n; i++) {
                c.set(new NDIndex(i), a.getFloat(i) + b.getFloat(i));
            }

            stopWatch.stop();
            log.info(String.format("%.5f sec", stopWatch.getTotalTimeSeconds()));

            log.info("=====================================");

            stopWatch.start();

            a.add(b);

            stopWatch.stop();
            log.info(String.format("%.5f sec", stopWatch.getTotalTimeSeconds()));
        }

    }
}
