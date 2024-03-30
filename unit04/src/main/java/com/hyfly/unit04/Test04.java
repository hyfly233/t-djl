package com.hyfly.unit04;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import ai.djl.util.Utils;
import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.nio.file.Files;

@Slf4j
public class Test04 {

    public static void main(String[] args) throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray x = manager.arange(4);
            try (FileOutputStream fos = new FileOutputStream("x-file")) {
                fos.write(x.encode());
            }
            log.info(x.toDebugString(true));

            //
            NDArray x2;
            try (FileInputStream fis = new FileInputStream("x-file")) {
                // We use the `Utils` method `toByteArray()` to read
                // from a `FileInputStream` and return it as a `byte[]`.
                x2 = NDArray.decode(manager, Utils.toByteArray(fis));
                log.info(x2.toDebugString(true));
            }

            //
            NDList list = new NDList(x, x2);
            try (FileOutputStream fos = new FileOutputStream("x-file")) {
                fos.write(list.encode());
            }
            try (FileInputStream fis = new FileInputStream("x-file")) {
                list = NDList.decode(manager, Utils.toByteArray(fis));
            }
            log.info(list.toString());

            //
            SequentialBlock original = createMLP();

            NDArray x1 = manager.randomUniform(0, 1, new Shape(2, 5));

            original.initialize(manager, DataType.FLOAT32, x1.getShape());

            ParameterStore ps = new ParameterStore(manager, false);
            NDArray y = original.forward(ps, new NDList(x1), false).singletonOrThrow();

            log.info(y.toDebugString(true));

            //
            File mlpParamFile = new File("mlp.param");
            DataOutputStream os = new DataOutputStream(Files.newOutputStream(mlpParamFile.toPath()));
            original.saveParameters(os);

            //
            // Create duplicate of network architecture
            SequentialBlock clone = createMLP();
            // Load Parameters
            clone.loadParameters(manager, new DataInputStream(Files.newInputStream(mlpParamFile.toPath())));

            // Original model's parameters
            PairList<String, Parameter> originalParams = original.getParameters();
            // Loaded model's parameters
            PairList<String, Parameter> loadedParams = clone.getParameters();

            for (int i = 0; i < originalParams.size(); i++) {
                if (originalParams.valueAt(i).getArray().equals(loadedParams.valueAt(i).getArray())) {
                    System.out.printf("True ");
                } else {
                    System.out.printf("False ");
                }
            }

            NDArray yClone = clone.forward(ps, new NDList(x), false).singletonOrThrow();

            y.equals(yClone);
        }
    }

    public static SequentialBlock createMLP() {
        SequentialBlock mlp = new SequentialBlock();
        mlp.add(Linear.builder().setUnits(256).build());
        mlp.add(Activation.reluBlock());
        mlp.add(Linear.builder().setUnits(10).build());
        return mlp;
    }

}
