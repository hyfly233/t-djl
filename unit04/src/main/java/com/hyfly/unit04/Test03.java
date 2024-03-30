package com.hyfly.unit04;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import com.hyfly.unit04.entity.CenteredLayer;
import com.hyfly.unit04.entity.MyLinear;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Test03 {

    public static void main(String[] args) throws TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            CenteredLayer layer = new CenteredLayer();

            Model model = Model.newInstance("centered-layer");
            model.setBlock(layer);

            Predictor<NDList, NDList> predictor = model.newPredictor(new NoopTranslator());
            NDArray input = manager.create(new float[]{1f, 2f, 3f, 4f, 5f});
            predictor.predict(new NDList(input)).singletonOrThrow();

            //
            SequentialBlock net = new SequentialBlock();
            net.add(Linear.builder().setUnits(128).build());
            net.add(new CenteredLayer());
            net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
            net.initialize(manager, DataType.FLOAT32, input.getShape());

            //
            NDArray input2 = manager.randomUniform(-0.07f, 0.07f, new Shape(4, 8));
            NDArray y = predictor.predict(new NDList(input2)).singletonOrThrow();
            NDArray mean = y.mean();
            log.info(mean.toDebugString(true));

            //
            // 5 units in -> 3 units out
            MyLinear linear = new MyLinear(3, 5);
            var params = linear.getParameters();
            for (Pair<String, Parameter> param : params) {
                System.out.println(param.getKey());
            }

            //
            NDArray input3 = manager.randomUniform(0, 1, new Shape(2, 5));

            linear.initialize(manager, DataType.FLOAT32, input3.getShape());

            Model model1 = Model.newInstance("my-linear");
            model1.setBlock(linear);

            Predictor<NDList, NDList> predictor1 = model1.newPredictor(new NoopTranslator());
            NDArray ndArray = predictor1.predict(new NDList(input3)).singletonOrThrow();

            log.info(ndArray.toDebugString(true));

            NDArray input4 = manager.randomUniform(0, 1, new Shape(2, 64));

            SequentialBlock net1 = new SequentialBlock();
            net1.add(new MyLinear(8, 64)); // 64 units in -> 8 units out
            net1.add(new MyLinear(1, 8)); // 8 units in -> 1 unit out
            net1.initialize(manager, DataType.FLOAT32, input4.getShape());

            Model model2 = Model.newInstance("lin-reg-custom");
            model2.setBlock(net1);

            Predictor<NDList, NDList> predictor2 = model2.newPredictor(new NoopTranslator());
            NDArray ndArray1 = predictor2.predict(new NDList(input4)).singletonOrThrow();

            log.info(ndArray1.toDebugString(true));
        }
    }
}
