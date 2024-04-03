package com.hyfly.unit07;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import com.hyfly.utils.PlotUtils;
import com.hyfly.utils.timemachine.SeqDataLoader;
import com.hyfly.utils.timemachine.TimeMachine;
import com.hyfly.utils.timemachine.Vocab;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Test03 {

    public static void main(String[] args) throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            String[][] tokens = TimeMachine.tokenize(TimeMachine.readTimeMachine(), "word");
            // 由于每一行文字不一定是一个句子或段落，因此我们把所有文本行拼接到一起
            List<String> corpus = new ArrayList<>();
            for (int i = 0; i < tokens.length; i++) {
                for (int j = 0; j < tokens[i].length; j++) {
                    if (tokens[i][j] != "") {
                        corpus.add(tokens[i][j]);
                    }
                }
            }

            Vocab vocab = new Vocab(new String[][]{corpus.toArray(new String[0])}, -1, new String[0]);
            for (int i = 0; i < 10; i++) {
                Map.Entry<String, Integer> token = vocab.tokenFreqs.get(i);
                System.out.println(token.getKey() + ": " + token.getValue());
            }

            int n = vocab.tokenFreqs.size();
            double[] freqs = new double[n];
            double[] x = new double[n];
            for (int i = 0; i < n; i++) {
                freqs[i] = (double) vocab.tokenFreqs.get(i).getValue();
                x[i] = (double) i;
            }

            PlotUtils.plotLogScale(new double[][]{x}, new double[][]{freqs}, new String[]{""},
                    "token: x", "frequency: n(x)");

            //
            String[] bigramTokens = new String[corpus.size() - 1];
            for (int i = 0; i < bigramTokens.length; i++) {
                bigramTokens[i] = corpus.get(i) + " " + corpus.get(i + 1);
            }
            Vocab bigramVocab = new Vocab(new String[][]{bigramTokens}, -1, new String[0]);
            for (int i = 0; i < 10; i++) {
                Map.Entry<String, Integer> token = bigramVocab.tokenFreqs.get(i);
                System.out.println(token.getKey() + ": " + token.getValue());
            }

            //
            String[] trigramTokens = new String[corpus.size() - 2];
            for (int i = 0; i < trigramTokens.length; i++) {
                trigramTokens[i] = corpus.get(i) + " " + corpus.get(i + 1) + " " + corpus.get(i + 2);
            }
            Vocab trigramVocab = new Vocab(new String[][]{trigramTokens}, -1, new String[0]);
            for (int i = 0; i < 10; i++) {
                Map.Entry<String, Integer> token = trigramVocab.tokenFreqs.get(i);
                System.out.println(token.getKey() + ": " + token.getValue());
            }

            //
            n = bigramVocab.tokenFreqs.size();
            double[] bigramFreqs = new double[n];
            double[] bigramX = new double[n];
            for (int i = 0; i < n; i++) {
                bigramFreqs[i] = (double) bigramVocab.tokenFreqs.get(i).getValue();
                bigramX[i] = (double) i;
            }

            n = trigramVocab.tokenFreqs.size();
            double[] trigramFreqs = new double[n];
            double[] trigramX = new double[n];
            for (int i = 0; i < n; i++) {
                trigramFreqs[i] = (double) trigramVocab.tokenFreqs.get(i).getValue();
                trigramX[i] = (double) i;
            }

            PlotUtils.plotLogScale(new double[][]{x, bigramX, trigramX}, new double[][]{freqs, bigramFreqs, trigramFreqs},
                    new String[]{"unigram", "bigram", "trigram"}, "token: x", "frequency: n(x)");

            //
            List<Integer> mySeq = new ArrayList<>();
            for (int i = 0; i < 35; i++) {
                mySeq.add(i);
            }

            for (NDList pair : SeqDataLoader.seqDataIterRandom(mySeq, 2, 5, manager)) {
                System.out.println("X:\n" + pair.get(0).toDebugString(50, 50, 50, 50, true));
                System.out.println("Y:\n" + pair.get(1).toDebugString(50, 50, 50, 50, true));
            }

            for (NDList pair : SeqDataLoader.seqDataIterSequential(mySeq, 2, 5, manager)) {
                System.out.println("X:\n" + pair.get(0).toDebugString(10, 10, 10, 10, true));
                System.out.println("Y:\n" + pair.get(1).toDebugString(10, 10, 10, 10, true));
            }
        }
    }
}
