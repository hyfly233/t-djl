package com.hyfly.unit08.entity;

import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.core.Linear;
import ai.djl.nn.recurrent.GRU;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.hyfly.utils.lstm.Decoder;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Seq2SeqDecoder extends Decoder {

    private TrainableWordEmbedding embedding;
    private GRU rnn;
    private Linear dense;

    /* The RNN decoder for sequence to sequence learning. */
    public Seq2SeqDecoder(
            int vocabSize, int embedSize, int numHiddens, int numLayers, float dropout) {
        List<String> list =
                IntStream.range(0, vocabSize)
                        .mapToObj(String::valueOf)
                        .collect(Collectors.toList());
        Vocabulary vocab = new DefaultVocabulary(list);
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
        dense = Linear.builder().setUnits(vocabSize).build();
        addChildBlock("dense", dense);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initializeChildBlocks(
            NDManager manager, DataType dataType, Shape... inputShapes) {
        embedding.initialize(manager, dataType, inputShapes[0]);
        try (NDManager sub = manager.newSubManager()) {
            Shape shape = embedding.getOutputShapes(new Shape[]{inputShapes[0]})[0];
            NDArray nd = sub.zeros(shape, dataType).swapAxes(0, 1);
            NDArray state = sub.zeros(inputShapes[1], dataType);
            NDArray context = state.get(new NDIndex(-1));
            context =
                    context.broadcast(
                            new Shape(
                                    nd.getShape().head(),
                                    context.getShape().head(),
                                    context.getShape().get(1)));
            // Broadcast `context` so it has the same `numSteps` as `X`
            NDArray xAndContext = NDArrays.concat(new NDList(nd, context), 2);
            rnn.initialize(manager, dataType, xAndContext.getShape());
            shape = rnn.getOutputShapes(new Shape[]{xAndContext.getShape()})[0];
            dense.initialize(manager, dataType, shape);
        }
    }

    public NDList initState(NDList encOutputs) {
        return new NDList(encOutputs.get(1));
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray X = inputs.head();
        NDArray state = inputs.get(1);
        X =
                embedding
                        .forward(parameterStore, new NDList(X), training, params)
                        .head()
                        .swapAxes(0, 1);
        NDArray context = state.get(new NDIndex(-1));
        // Broadcast `context` so it has the same `numSteps` as `X`
        context =
                context.broadcast(
                        new Shape(
                                X.getShape().head(),
                                context.getShape().head(),
                                context.getShape().get(1)));
        NDArray xAndContext = NDArrays.concat(new NDList(X, context), 2);
        NDList rnnOutput =
                rnn.forward(parameterStore, new NDList(xAndContext, state), training);
        NDArray output = rnnOutput.head();
        state = rnnOutput.get(1);
        output =
                dense.forward(parameterStore, new NDList(output), training)
                        .head()
                        .swapAxes(0, 1);
        return new NDList(output, state);
    }
}
