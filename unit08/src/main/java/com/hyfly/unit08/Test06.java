package com.hyfly.unit08;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import com.hyfly.unit08.entity.MaskedSoftmaxCELoss;
import com.hyfly.unit08.entity.Seq2SeqDecoder;
import com.hyfly.unit08.entity.Seq2SeqEncoder;
import com.hyfly.utils.*;
import com.hyfly.utils.lstm.EncoderDecoder;
import com.hyfly.utils.timemachine.Vocab;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.stream.Stream;

public class Test06 {

    public static void main(String[] args) throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            ParameterStore ps = new ParameterStore(manager, false);

            Seq2SeqEncoder encoder = new Seq2SeqEncoder(10, 8, 16, 2, 0);
            NDArray X = manager.zeros(new Shape(4, 7));
            encoder.initialize(manager, DataType.FLOAT32, X.getShape());
            NDList outputState = encoder.forward(ps, new NDList(X), false);
            NDArray output = outputState.head();

            output.getShape();

            NDList state = outputState.subNDList(1);
            System.out.println(state.size());
            System.out.println(state.head().getShape());

            //
            Seq2SeqDecoder decoder = new Seq2SeqDecoder(10, 8, 16, 2, 0);
            state = decoder.initState(outputState);
            NDList input = new NDList(X).addAll(state);
            decoder.initialize(manager, DataType.FLOAT32, input.getShapes());
            outputState = decoder.forward(ps, input, false);

            output = outputState.head();
            System.out.println(output.getShape());

            state = outputState.subNDList(1);
            System.out.println(state.size());
            System.out.println(state.head().getShape());

            //
            X = manager.create(new int[][]{{1, 2, 3}, {4, 5, 6}});
            System.out.println(X.sequenceMask(manager.create(new int[]{1, 2})));

            //
            X = manager.ones(new Shape(2, 3, 4));
            System.out.println(X.sequenceMask(manager.create(new int[]{1, 2}), -1));

            //
            Loss loss = new MaskedSoftmaxCELoss();
            NDList labels = new NDList(manager.ones(new Shape(3, 4)));
            labels.add(manager.create(new int[]{4, 2, 0}));
            NDList predictions = new NDList(manager.ones(new Shape(3, 4, 10)));
            System.out.println(loss.evaluate(labels, predictions));

            //
            int embedSize = 32;
            int numHiddens = 32;
            int numLayers = 2;
            int batchSize = 64;
            int numSteps = 10;
            int numEpochs = Integer.getInteger("MAX_EPOCH", 300);

            float dropout = 0.1f, lr = 0.005f;
            Device device = manager.getDevice();

            Pair<ArrayDataset, Pair<Vocab, Vocab>> dataNMT =
                    NMT.loadDataNMT(batchSize, numSteps, 600, manager);
            ArrayDataset dataset = dataNMT.getKey();
            Vocab srcVocab = dataNMT.getValue().getKey();
            Vocab tgtVocab = dataNMT.getValue().getValue();

            encoder = new Seq2SeqEncoder(srcVocab.length(), embedSize, numHiddens, numLayers, dropout);
            decoder = new Seq2SeqDecoder(tgtVocab.length(), embedSize, numHiddens, numLayers, dropout);

            EncoderDecoder net = new EncoderDecoder(encoder, decoder);
            trainSeq2Seq(manager, net, dataset, lr, numEpochs, tgtVocab, device);

            //
            String[] engs = {"go .", "i lost .", "he\'s calm .", "i\'m home ."};
            String[] fras = {"va !", "j\'ai perdu .", "il est calme .", "je suis chez moi ."};
            for (int i = 0; i < engs.length; i++) {
                Pair<String, ArrayList<NDArray>> pair = predictSeq2Seq(manager, net, engs[i], srcVocab, tgtVocab, numSteps, device, false);
                String translation = pair.getKey();
                ArrayList<NDArray> attentionWeightSeq = pair.getValue();
                System.out.format("%s => %s, bleu %.3f\n", engs[i], translation, bleu(translation, fras[i], 2));
            }
        }
    }

    public static void trainSeq2Seq(
            NDManager manager,
            EncoderDecoder net,
            ArrayDataset dataset,
            float lr,
            int numEpochs,
            Vocab tgtVocab,
            Device device)
            throws IOException, TranslateException {
        Loss loss = new MaskedSoftmaxCELoss();
        Tracker lrt = Tracker.fixed(lr);
        Optimizer adam = Optimizer.adam().optLearningRateTracker(lrt).build();

        DefaultTrainingConfig config =
                new DefaultTrainingConfig(loss)
                        .optOptimizer(adam) // Optimizer (loss function)
                        .optInitializer(new XavierInitializer(), "");

        Model model = Model.newInstance("");
        model.setBlock(net);
        Trainer trainer = model.newTrainer(config);

        Animator animator = new Animator();
        StopWatch watch;
        Accumulator metric;
        double lossValue = 0, speed = 0;
        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            watch = new StopWatch();
            metric = new Accumulator(2); // Sum of training loss, no. of tokens
            try (NDManager childManager = manager.newSubManager(device)) {
                // Iterate over dataset
                for (Batch batch : dataset.getData(childManager)) {
                    NDArray X = batch.getData().get(0);
                    NDArray lenX = batch.getData().get(1);
                    NDArray Y = batch.getLabels().get(0);
                    NDArray lenY = batch.getLabels().get(1);

                    NDArray bos =
                            childManager
                                    .full(new Shape(Y.getShape().get(0)), tgtVocab.getIdx("<bos>"))
                                    .reshape(-1, 1);
                    NDArray decInput =
                            NDArrays.concat(
                                    new NDList(bos, Y.get(new NDIndex(":, :-1"))),
                                    1); // Teacher forcing
                    try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                        NDArray yHat =
                                net.forward(
                                                new ParameterStore(manager, false),
                                                new NDList(X, decInput, lenX),
                                                true)
                                        .get(0);
                        NDArray l = loss.evaluate(new NDList(Y, lenY), new NDList(yHat));
                        gc.backward(l);
                        metric.add(new float[]{l.sum().getFloat(), lenY.sum().getLong()});
                    }
                    TrainingChapter9.gradClipping(net, 1, childManager);
                    // Update parameters
                    trainer.step();
                }
            }
            lossValue = metric.get(0) / metric.get(1);
            speed = metric.get(1) / watch.stop();
            if ((epoch + 1) % 10 == 0) {
                animator.add(epoch + 1, (float) lossValue, "loss");
                animator.show();
            }
        }
        System.out.format(
                "loss: %.3f, %.1f tokens/sec on %s%n", lossValue, speed, device.toString());
    }

    /* 序列到序列模型的预测 */
    public static Pair<String, ArrayList<NDArray>> predictSeq2Seq(
            NDManager manager,
            EncoderDecoder net,
            String srcSentence,
            Vocab srcVocab,
            Vocab tgtVocab,
            int numSteps,
            Device device,
            boolean saveAttentionWeights)
            throws IOException, TranslateException {
        Integer[] srcTokens =
                Stream.concat(
                                Arrays.stream(
                                        srcVocab.getIdxs(srcSentence.toLowerCase().split(" "))),
                                Arrays.stream(new Integer[]{srcVocab.getIdx("<eos>")}))
                        .toArray(Integer[]::new);
        NDArray encValidLen = manager.create(srcTokens.length);
        int[] truncateSrcTokens = NMT.truncatePad(srcTokens, numSteps, srcVocab.getIdx("<pad>"));
        // 添加批量轴
        NDArray encX = manager.create(truncateSrcTokens).expandDims(0);
        NDList encOutputs =
                net.getEncoder().forward(
                        new ParameterStore(manager, false), new NDList(encX, encValidLen), false);
        NDList decState = net.getDecoder().initState(encOutputs.addAll(new NDList(encValidLen)));
        // 添加批量轴
        NDArray decX = manager.create(new float[]{tgtVocab.getIdx("<bos>")}).expandDims(0);
        ArrayList<Integer> outputSeq = new ArrayList<>();
        ArrayList<NDArray> attentionWeightSeq = new ArrayList<>();
        for (int i = 0; i < numSteps; i++) {
            NDList output =
                    net.getDecoder().forward(
                            new ParameterStore(manager, false),
                            new NDList(decX).addAll(decState),
                            false);
            NDArray Y = output.get(0);
            decState = output.subNDList(1);
            // 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
            decX = Y.argMax(2);
            int pred = (int) decX.squeeze(0).getLong();
            // 保存注意力权重（稍后讨论）
            if (saveAttentionWeights) {
                attentionWeightSeq.add(net.getDecoder().getAttentionWeights());
            }
            // 一旦序列结束词元被预测，输出序列的生成就完成了
            if (pred == tgtVocab.getIdx("<eos>")) {
                break;
            }
            outputSeq.add(pred);
        }
        String outputString =
                String.join(" ", tgtVocab.toTokens(outputSeq).toArray(new String[]{}));
        return new Pair<>(outputString, attentionWeightSeq);
    }

    /* 计算 BLEU. */
    public static double bleu(String predSeq, String labelSeq, int k) {
        String[] predTokens = predSeq.split(" ");
        String[] labelTokens = labelSeq.split(" ");
        int lenPred = predTokens.length;
        int lenLabel = labelTokens.length;
        double score = Math.exp(Math.min(0, 1 - lenLabel / lenPred));
        for (int n = 1; n < k + 1; n++) {
            float numMatches = 0f;
            HashMap<String, Integer> labelSubs = new HashMap<>();
            for (int i = 0; i < lenLabel - n + 1; i++) {
                String key =
                        String.join(" ", Arrays.copyOfRange(labelTokens, i, i + n, String[].class));
                labelSubs.put(key, labelSubs.getOrDefault(key, 0) + 1);
            }
            for (int i = 0; i < lenPred - n + 1; i++) {
                String key =
                        String.join(" ", Arrays.copyOfRange(predTokens, i, i + n, String[].class));
                if (labelSubs.getOrDefault(key, 0) > 0) {
                    numMatches += 1;
                    labelSubs.put(key, labelSubs.getOrDefault(key, 0) - 1);
                }
            }
            score *= Math.pow(numMatches / (lenPred - n + 1), Math.pow(0.5, n));
        }
        return score;
    }
}
