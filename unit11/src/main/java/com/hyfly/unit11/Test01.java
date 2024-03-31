package com.hyfly.unit11;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.hyfly.utils.StopWatch;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;

@Slf4j
public class Test01 {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray x_cpu = manager.randomUniform(0f, 1f, new Shape(2000, 2000), DataType.FLOAT32, Device.cpu());
            NDArray x_gpu = manager.randomUniform(0f, 1f, new Shape(6000, 6000), DataType.FLOAT32, Device.gpu());

            // 设备初始化预热
            run(x_cpu);
            run(x_gpu);

            // 计算CPU计算时间
            StopWatch stopWatch0 = new StopWatch();
            stopWatch0.start();

            run(x_cpu);

            stopWatch0.stop();
            ArrayList<Double> times = stopWatch0.getTimes();
            System.out.println("CPU time: " + times.get(times.size() - 1) + " nanoseconds ");

            // 计算GPU计算时间
            StopWatch stopWatch1 = new StopWatch();
            stopWatch1.start();

            run(x_gpu);

            stopWatch1.stop();
            times = stopWatch1.getTimes();
            System.out.println("GPU time: " + times.get(times.size() - 1) + " nanoseconds ");

            // 计算CPU和GPU的组合计算时间
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();

            run(x_cpu);
            run(x_gpu);

            stopWatch.stop();
            times = stopWatch.getTimes();
            System.out.println("CPU & GPU: " + times.get(times.size() - 1) + " nanoseconds ");

            // 计算GPU计算时间
            stopWatch = new StopWatch();
            stopWatch.start();

            NDArray Y = run(x_gpu);

            stopWatch.stop();
            times = stopWatch.getTimes();
            System.out.println("Run on GPU: " + times.get(times.size() - 1) + " nanoseconds ");

            // 计算复制到CPU的时间
            stopWatch1 = new StopWatch();
            stopWatch1.start();

            NDArray y_cpu = copyToCPU(Y);

            stopWatch1.stop();
            times = stopWatch1.getTimes();
            System.out.println("Copy to CPU: " + times.get(times.size() - 1) + " nanoseconds ");

            // 计算组合GPU计算和复制到CPU时间。
            stopWatch = new StopWatch();
            stopWatch.start();

            Y = run(x_gpu);
            y_cpu = copyToCPU(Y);

            stopWatch.stop();
            times = stopWatch.getTimes();
            System.out.println("Run on GPU and copy to CPU: " + times.get(times.size() - 1) + " nanoseconds ");
        }
    }

    public static NDArray run(NDArray X) {

        for (int i = 0; i < 10; i++) {
            X = X.dot(X);
        }
        return X;
    }

    public static NDArray copyToCPU(NDArray X) {
        return X.toDevice(Device.cpu(), true);
    }
}
