package com.hyfly.unit08.entity;

import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.recurrent.GRU;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.hyfly.utils.lstm.Encoder;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Seq2SeqEncoder extends Encoder {

    private TrainableWordEmbedding embedding;
    private GRU rnn;

    // 用于序列到序列学习的循环神经网络编码器
    public Seq2SeqEncoder(
            int vocabSize, int embedSize, int numHiddens, int numLayers, float dropout) {
        List<String> list =
                IntStream.range(0, vocabSize)
                        .mapToObj(String::valueOf)
                        .collect(Collectors.toList());
        Vocabulary vocab = new DefaultVocabulary(list);
        // Embedding layer
        embedding =
                TrainableWordEmbedding.builder()
                        .optNumEmbeddings(vocabSize)
                        .setEmbeddingSize(embedSize)
                        .setVocabulary(vocab)
                        .build();
        addChildBlock("embedding", embedding);
        rnn =
                GRU.builder()
                        .setNumLayers(numLayers)
                        .setStateSize(numHiddens)
                        .optReturnState(true)
                        .optBatchFirst(false)
                        .optDropRate(dropout)
                        .build();
        addChildBlock("rnn", rnn);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initializeChildBlocks(
            NDManager manager, DataType dataType, Shape... inputShapes) {
        embedding.initialize(manager, dataType, inputShapes[0]);
        Shape[] shapes = embedding.getOutputShapes(new Shape[]{inputShapes[0]});
        try (NDManager sub = manager.newSubManager()) {
            NDArray nd = sub.zeros(shapes[0], dataType);
            nd = nd.swapAxes(0, 1);
            rnn.initialize(manager, dataType, nd.getShape());
        }
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore ps,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray X = inputs.head();
        // 输出'X'的形状: (batchSize, numSteps, embedSize)
        X = embedding.forward(ps, new NDList(X), training, params).head();
        // 在循环神经网络模型中，第一个轴对应于时间步
        X = X.swapAxes(0, 1);

        return rnn.forward(ps, new NDList(X), training);
    }
}
