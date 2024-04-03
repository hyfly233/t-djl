package com.hyfly.unit07;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.training.GradientCollector;
import ai.djl.training.loss.Loss;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.util.Pair;
import com.hyfly.unit07.entity.RNNModelScratch;
import com.hyfly.utils.*;
import com.hyfly.utils.Functions.TriFunction;
import com.hyfly.utils.timemachine.SeqDataLoader;
import com.hyfly.utils.timemachine.Vocab;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;


public class Test05 {

    public static void main(String[] args) throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            int batchSize = 32;
            int numSteps = 35;
            Pair<List<NDList>, Vocab> timeMachine = SeqDataLoader.loadDataTimeMachine(batchSize, numSteps, false, 10000, manager);
            List<NDList> trainIter = timeMachine.getKey();
            Vocab vocab = timeMachine.getValue();

            manager.create(new int[]{0, 2}).oneHot(vocab.length());

            NDArray X = manager.arange(10).reshape(new Shape(2, 5));
            X.transpose().oneHot(28).getShape();

            //
            int numHiddens = 512;
            Functions.TriFunction<Integer, Integer, Device, NDList> getParamsFn = (a, b, c) -> getParams(manager, a, b, c);
            TriFunction<Integer, Integer, Device, NDList> initRNNStateFn =
                    (a, b, c) -> initRNNState(manager, a, b, c);
            TriFunction<NDArray, NDList, NDList, Pair> rnnFn = (a, b, c) -> rnn(a, b, c);

            X = manager.arange(10).reshape(new Shape(2, 5));
            Device device = manager.getDevice();

            RNNModelScratch net =
                    new RNNModelScratch(
                            vocab.length(), numHiddens, device, getParamsFn, initRNNStateFn, rnnFn);
            NDList state = net.beginState((int) X.getShape().getShape()[0], device);
            Pair<NDArray, NDList> pairResult = net.forward(X.toDevice(device, false), state);
            NDArray Y = pairResult.getKey();
            NDList newState = pairResult.getValue();
            System.out.println(Y.getShape());
            System.out.println(newState.get(0).getShape());

            //
            predictCh8(manager, "time traveller ", 10, net, vocab, manager.getDevice());

            //
            int numEpochs = Integer.getInteger("MAX_EPOCH", 500);

            int lr = 1;
            trainCh8(manager, net, trainIter, vocab, lr, numEpochs, manager.getDevice(), false);

            //
            trainCh8(manager, net, trainIter, vocab, lr, numEpochs, manager.getDevice(), true);
        }
    }

    public static NDList initRNNState(NDManager manager, int batchSize, int numHiddens, Device device) {
        return new NDList(manager.zeros(new Shape(batchSize, numHiddens), DataType.FLOAT32, device));
    }

    public static Pair<NDArray, NDList> rnn(NDArray inputs, NDList state, NDList params) {
        // 输入的形状：（`numSteps`、`batchSize`、`vocabSize`）
        NDArray W_xh = params.get(0);
        NDArray W_hh = params.get(1);
        NDArray b_h = params.get(2);
        NDArray W_hq = params.get(3);
        NDArray b_q = params.get(4);
        NDArray H = state.get(0);

        NDList outputs = new NDList();
        // 'X'的形状：（'batchSize'，'vocabSize`）
        NDArray X, Y;
        for (int i = 0; i < inputs.size(0); i++) {
            X = inputs.get(i);
            H = (X.dot(W_xh).add(H.dot(W_hh)).add(b_h)).tanh();
            Y = H.dot(W_hq).add(b_q);
            outputs.add(Y);
        }
        return new Pair<>(outputs.size() > 1 ? NDArrays.concat(outputs) : outputs.get(0), new NDList(H));
    }

    public static NDArray normal(NDManager manager, Shape shape, Device device) {
        return manager.randomNormal(0, 0.01f, shape, DataType.FLOAT32, device);
    }

    public static NDList three(NDManager manager, int numInputs, int numHiddens, Device device) {
        return new NDList(
                normal(manager, new Shape(numInputs, numHiddens), device),
                normal(manager, new Shape(numHiddens, numHiddens), device),
                manager.zeros(new Shape(numHiddens), DataType.FLOAT32, device));
    }

    public static NDList getParams(NDManager manager, int vocabSize, int numHiddens, Device device) {
        int numInputs = vocabSize;
        int numOutputs = vocabSize;

        // Update gate parameters
        NDList temp = three(manager, numInputs, numHiddens, device);
        NDArray W_xz = temp.get(0);
        NDArray W_hz = temp.get(1);
        NDArray b_z = temp.get(2);

        // Reset gate parameters
        temp = three(manager, numInputs, numHiddens, device);
        NDArray W_xr = temp.get(0);
        NDArray W_hr = temp.get(1);
        NDArray b_r = temp.get(2);

        // Candidate hidden state parameters
        temp = three(manager, numInputs, numHiddens, device);
        NDArray W_xh = temp.get(0);
        NDArray W_hh = temp.get(1);
        NDArray b_h = temp.get(2);

        // Output layer parameters
        NDArray W_hq = normal(manager, new Shape(numHiddens, numOutputs), device);
        NDArray b_q = manager.zeros(new Shape(numOutputs), DataType.FLOAT32, device);

        // Attach gradients
        NDList params = new NDList(W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q);
        for (NDArray param : params) {
            param.setRequiresGradient(true);
        }
        return params;
    }

    public static NDList initGruState(NDManager manager, int batchSize, int numHiddens, Device device) {
        return new NDList(manager.zeros(new Shape(batchSize, numHiddens), DataType.FLOAT32, device));
    }

    public static Pair<NDArray, NDList> gru(NDArray inputs, NDList state, NDList params) {
        NDArray W_xz = params.get(0);
        NDArray W_hz = params.get(1);
        NDArray b_z = params.get(2);

        NDArray W_xr = params.get(3);
        NDArray W_hr = params.get(4);
        NDArray b_r = params.get(5);

        NDArray W_xh = params.get(6);
        NDArray W_hh = params.get(7);
        NDArray b_h = params.get(8);

        NDArray W_hq = params.get(9);
        NDArray b_q = params.get(10);

        NDArray H = state.get(0);
        NDList outputs = new NDList();
        NDArray X, Y, Z, R, H_tilda;
        for (int i = 0; i < inputs.size(0); i++) {
            X = inputs.get(i);
            Z = Activation.sigmoid(X.dot(W_xz).add(H.dot(W_hz).add(b_z)));
            R = Activation.sigmoid(X.dot(W_xr).add(H.dot(W_hr).add(b_r)));
            H_tilda = Activation.tanh(X.dot(W_xh).add(R.mul(H).dot(W_hh).add(b_h)));
            H = Z.mul(H).add(Z.mul(-1).add(1).mul(H_tilda));
            Y = H.dot(W_hq).add(b_q);
            outputs.add(Y);
        }
        return new Pair(outputs.size() > 1 ? NDArrays.concat(outputs) : outputs.get(0), new NDList(H));
    }

    /**
     * 在 `prefix` 后面生成新字符。
     */
    public static String predictCh8(NDManager manager,
                                    String prefix, int numPreds, RNNModelScratch net, Vocab vocab, Device device) {
        NDList state = net.beginState(1, device);
        List<Integer> outputs = new ArrayList<>();
        outputs.add(vocab.getIdx("" + prefix.charAt(0)));
        Functions.SimpleFunction<NDArray> getInput =
                () ->
                        manager.create(outputs.get(outputs.size() - 1))
                                .toDevice(device, false)
                                .reshape(new Shape(1, 1));
        for (char c : prefix.substring(1).toCharArray()) { // 预热期
            state = (NDList) net.forward(getInput.apply(), state).getValue();
            outputs.add(vocab.getIdx("" + c));
        }

        NDArray y;
        for (int i = 0; i < numPreds; i++) {
            Pair<NDArray, NDList> pair = net.forward(getInput.apply(), state);
            y = pair.getKey();
            state = pair.getValue();

            outputs.add((int) y.argMax(1).reshape(new Shape(1)).getLong(0L));
        }
        StringBuilder output = new StringBuilder();
        for (int i : outputs) {
            output.append(vocab.idxToToken.get(i));
        }
        return output.toString();
    }

    /**
     * 修剪梯度
     */
    public static void gradClipping(RNNModelScratch net, int theta, NDManager manager) {
        double result = 0;
        for (NDArray p : net.params) {
            NDArray gradient = p.getGradient();
            gradient.attach(manager);
            result += gradient.pow(2).sum().getFloat();
        }
        double norm = Math.sqrt(result);
        if (norm > theta) {
            for (NDArray param : net.params) {
                NDArray gradient = param.getGradient();
                gradient.muli(theta / norm);
            }
        }
    }

    /**
     * 在一个opoch内训练一个模型。
     */
    public static Pair<Double, Double> trainEpochCh8(NDManager manager,
                                                     RNNModelScratch net,
                                                     List<NDList> trainIter,
                                                     Loss loss,
                                                     Functions.voidTwoFunction<Integer, NDManager> updater,
                                                     Device device,
                                                     boolean useRandomIter) {
        StopWatch watch = new StopWatch();
        watch.start();
        Accumulator metric = new Accumulator(2); // 训练损失总数
        try (NDManager childManager = manager.newSubManager()) {
            NDList state = null;
            for (NDList pair : trainIter) {
                NDArray X = pair.get(0).toDevice(device, true);
                X.attach(childManager);
                NDArray Y = pair.get(1).toDevice(device, true);
                Y.attach(childManager);
                if (state == null || useRandomIter) {
                    // 在第一次迭代或
                    // 使用随机取样
                    state = net.beginState((int) X.getShape().getShape()[0], device);
                } else {
                    for (NDArray s : state) {
                        s.stopGradient();
                    }
                }
                state.attach(childManager);

                NDArray y = Y.transpose().reshape(new Shape(-1));
                X = X.toDevice(device, false);
                y = y.toDevice(device, false);
                try (GradientCollector gc = manager.getEngine().newGradientCollector()) {
                    Pair<NDArray, NDList> pairResult = net.forward(X, state);
                    NDArray yHat = pairResult.getKey();
                    state = pairResult.getValue();
                    NDArray l = loss.evaluate(new NDList(y), new NDList(yHat)).mean();
                    gc.backward(l);
                    metric.add(new float[]{l.getFloat() * y.size(), y.size()});
                }
                gradClipping(net, 1, childManager);
                updater.apply(1, childManager); // 因为已经调用了“mean”函数
            }
        }
        return new Pair<>(Math.exp(metric.get(0) / metric.get(1)), metric.get(1) / watch.stop());
    }

    /**
     * 训练一个模型
     */
    public static void trainCh8(NDManager manager,
                                RNNModelScratch net,
                                List<NDList> trainIter,
                                Vocab vocab,
                                int lr,
                                int numEpochs,
                                Device device,
                                boolean useRandomIter) {
        SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss();
        Animator animator = new Animator();
        // 初始化
        Functions.voidTwoFunction<Integer, NDManager> updater =
                (batchSize, subManager) -> Training.sgd(net.params, lr, batchSize, subManager);
        Function<String, String> predict = (prefix) -> predictCh8(manager, prefix, 50, net, vocab, device);
        // 训练和推理
        double ppl = 0.0;
        double speed = 0.0;
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            Pair<Double, Double> pair =
                    trainEpochCh8(manager, net, trainIter, loss, updater, device, useRandomIter);
            ppl = pair.getKey();
            speed = pair.getValue();
            if ((epoch + 1) % 10 == 0) {
                animator.add(epoch + 1, (float) ppl, "");
                animator.show();
            }
        }
        System.out.format(
                "perplexity: %.1f, %.1f tokens/sec on %s%n", ppl, speed, device.toString());
        System.out.println(predict.apply("time traveller"));
        System.out.println(predict.apply("traveller"));
    }
}
